from __future__ import annotations

import hmac

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.params import Query

from app.core.config import API_PREFIX, CALENDAR_TOOL_NAME, DEMO_USER_ID, logger, settings
from app.models.schemas import (
    BoundingBox,
    CallEndedPayload,
    CallEndedResponse,
    CallReportItem,
    ContextResponse,
    DetectedPerson,
    FaceRegisterPayload,
    KnownFacesResponse,
    MeetingStartPayload,
    MeetingTriggerResponse,
    MemoriesResponse,
    MemoryMatch,
    ReportsResponse,
    StatusMessageResponse,
    ToolResultModel,
    TranscriptResponse,
    VapiCallItem,
    VapiCallsResponse,
    VapiInjectPayload,
    VapiInjectResponse,
    VapiProvisionPayload,
    VapiProvisionResponse,
    VisionCapturePayload,
    VisionDetectPayload,
    VisionDetectResponse,
    WebhookPayload,
    WebhookResponse,
)
from app.services import memory_agent, vapi_client, face_service

router = APIRouter(prefix=API_PREFIX)

_PROACTIVE_TRIGGER_CATEGORIES = {"task", "meeting", "commitment"}


def _build_proactive_system_prompt(memory_texts: list[str]) -> str:
    memory_context = ", ".join(memory_texts) if memory_texts else "aucun"
    return (
        "Tu es Memento, un copilote proactif avec mémoire persistante. "
        "Tu ne te comportes pas comme un simple moteur de questions-réponses. "
        "N'attends pas une question explicite pour aider. "
        "Dès qu'un sujet est évoqué, croise immédiatement avec la mémoire et propose du contexte pertinent. "
        "Si une échéance, un engagement, un risque ou un suivi est détecté, préviens l'utilisateur tout de suite et suggère la prochaine action concrète. "
        "Pour une alerte proactive, adopte un ton légèrement urgent, ferme et direct. "
        "Pour les alertes proactives, commence toujours la sortie orale par la phrase exacte « Alerte système : ». "
        "Sois proactif sur ces sujets. "
        "Contexte connu : "
        f"{memory_context}."
    )


def _build_memory_saved_result(category: str) -> str:
    normalized = (category or "").strip().lower()
    if normalized not in _PROACTIVE_TRIGGER_CATEGORIES:
        return "Mémoire enregistrée avec succès."

    return (
        "Mémoire enregistrée avec succès. "
        "Directive interne Memento : demande proactivement à l'utilisateur s'il souhaite planifier un suivi pour cet engagement maintenant. "
        "Protocole interne Memento : tu déclenches une alerte proactive. "
        "Tu DOIS adopter un ton légèrement plus urgent, ferme et direct. "
        "Préfixe strictement ta réponse orale par la phrase « Alerte système : »."
    )


async def _verify_webhook_token(
    x_webhook_token: str | None = Header(default=None, alias="X-Webhook-Token"),
) -> None:
    configured = (settings.webhook_token or "").strip()
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="Webhook auth is not configured on this server.",
        )

    token = (x_webhook_token or "").strip()
    if not token or not hmac.compare_digest(token, configured):
        raise HTTPException(status_code=401, detail="Unauthorized webhook request")


def _internal_error(detail: str, exc: Exception) -> None:
    logger.exception(detail, exc_info=exc)
    raise HTTPException(status_code=500, detail=detail)


@router.get("/transcript", response_model=TranscriptResponse)
async def get_transcript() -> TranscriptResponse:
    try:
        with memory_agent._transcript_lock:
            messages = list(memory_agent._transcript)
        return TranscriptResponse(status="success", messages=messages)
    except Exception as exc:
        _internal_error("Failed to fetch transcript", exc)


@router.delete("/transcript", response_model=StatusMessageResponse)
async def clear_transcript() -> StatusMessageResponse:
    try:
        with memory_agent._transcript_lock:
            memory_agent._transcript.clear()
        return StatusMessageResponse(status="success", message="Flux neural effacé.")
    except Exception as exc:
        _internal_error("Failed to clear transcript", exc)


@router.get("/memories", response_model=MemoriesResponse)
async def get_memories() -> MemoriesResponse:
    try:
        client = memory_agent._get_mem0_client()
        try:
            raw_memories = await run_in_threadpool(
                client.get_all,
                filters={"user_id": DEMO_USER_ID},
            )
        except Exception:
            raw_memories = await run_in_threadpool(client.get_all, user_id=DEMO_USER_ID)
        normalized = memory_agent._normalize_memories(raw_memories)
        return MemoriesResponse(status="success", data=normalized)
    except HTTPException:
        raise
    except Exception as exc:
        _internal_error("Failed to fetch memories", exc)


@router.get("/user/context", response_model=ContextResponse)
async def get_user_context() -> ContextResponse:
    fallback_prompt = _build_proactive_system_prompt(memory_texts=[])

    try:
        client = memory_agent._get_mem0_client()
        try:
            raw_memories = await run_in_threadpool(
                client.get_all,
                filters={"user_id": DEMO_USER_ID},
            )
        except Exception:
            raw_memories = await run_in_threadpool(client.get_all, user_id=DEMO_USER_ID)

        normalized = memory_agent._normalize_memories(raw_memories)
        memory_texts = [
            f"[{memory_agent._sanitize_preview(item.text, 120)}]"
            for item in normalized
            if item.text
        ]

        if not memory_texts:
            return ContextResponse(status="success", injected_prompt=fallback_prompt)

        injected_prompt = _build_proactive_system_prompt(memory_texts=memory_texts)
        return ContextResponse(status="success", injected_prompt=injected_prompt)
    except Exception:
        return ContextResponse(status="success", injected_prompt=fallback_prompt)


@router.post(
    "/webhook/save_memory",
    response_model=WebhookResponse,
    dependencies=[Depends(_verify_webhook_token)],
)
async def save_memory_webhook(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
) -> WebhookResponse:
    try:
        tool_call_id, fact, category = memory_agent._extract_first_valid_tool_call(payload)
        if not tool_call_id or not fact:
            return WebhookResponse(results=[])

        preview = memory_agent._sanitize_preview(fact)
        memory_agent._append_transcript("assistant", f"[Mémoire enregistrée : {preview}]")

        # Network write is offloaded to keep webhook latency below strict limits.
        background_tasks.add_task(memory_agent._save_memory_in_background, fact, category)

        if (category or "").strip().lower() in _PROACTIVE_TRIGGER_CATEGORIES:
            memory_agent._append_transcript(
                "assistant",
                "[ACTION PROACTIVE MEMENTO : engagement détecté — proposez de planifier un suivi maintenant.]",
            )

        return WebhookResponse(
            results=[
                ToolResultModel(
                    toolCallId=tool_call_id,
                    result=_build_memory_saved_result(category=category),
                )
            ]
        )
    except HTTPException:
        raise
    except Exception as exc:
        _internal_error("Failed to persist memory", exc)


@router.post(
    "/webhook/check_calendar",
    response_model=WebhookResponse,
    dependencies=[Depends(_verify_webhook_token)],
)
async def check_calendar_webhook(payload: WebhookPayload) -> WebhookResponse:
    try:
        tool_call_id = memory_agent._extract_tool_call_id(payload, CALENDAR_TOOL_NAME)
        if not tool_call_id:
            return WebhookResponse(results=[])

        result = "Agenda du jour : 14h00 — réunion du conseil, 16h00 — rendez-vous dentiste."
        memory_agent._append_transcript("assistant", "[Agenda consulté : 2 événements trouvés]")

        return WebhookResponse(
            results=[
                ToolResultModel(
                    toolCallId=tool_call_id,
                    result=result,
                )
            ]
        )
    except HTTPException:
        raise
    except Exception as exc:
        _internal_error("Failed to read calendar", exc)


@router.post(
    "/webhook/call_ended",
    response_model=CallEndedResponse,
    dependencies=[Depends(_verify_webhook_token)],
)
async def call_ended_webhook(payload: CallEndedPayload) -> CallEndedResponse:
    try:
        data = payload.root if isinstance(payload.root, dict) else {}
        transcript_excerpt = memory_agent._extract_transcript_excerpt(data.get("transcript"))
        summary = "Call completed. 3 action items identified."

        report = CallReportItem(
            timestamp=memory_agent._utc_now_iso(),
            summary=summary,
            transcript_excerpt=transcript_excerpt,
        )
        with memory_agent._reports_lock:
            memory_agent._call_reports.insert(0, report)

        return CallEndedResponse(status="success", report=summary)
    except Exception:
        fallback = "Call completed. Report generation deferred."
        report = CallReportItem(
            timestamp=memory_agent._utc_now_iso(),
            summary=fallback,
            transcript_excerpt=None,
        )
        with memory_agent._reports_lock:
            memory_agent._call_reports.insert(0, report)
        return CallEndedResponse(status="success", report=fallback)


@router.get("/reports", response_model=ReportsResponse)
async def get_reports() -> ReportsResponse:
    with memory_agent._reports_lock:
        reports = list(memory_agent._call_reports)
    return ReportsResponse(status="success", reports=reports)


@router.delete("/memories/forget", response_model=StatusMessageResponse)
async def forget_memories() -> StatusMessageResponse:
    try:
        client = memory_agent._get_mem0_client()

        try:
            await run_in_threadpool(client.delete_all, filters={"user_id": DEMO_USER_ID})
        except Exception:
            try:
                await run_in_threadpool(client.delete_all, user_id=DEMO_USER_ID)
            except Exception:
                await run_in_threadpool(client.delete, user_id=DEMO_USER_ID)

        return StatusMessageResponse(
            status="success",
            message="Toutes les mémoires ont été supprimées. État réinitialisé.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _internal_error("Failed to forget memories", exc)


@router.post("/vision/capture", response_model=StatusMessageResponse)
async def capture_vision_context(
    payload: VisionCapturePayload,
    background_tasks: BackgroundTasks,
) -> StatusMessageResponse:
    try:
        description = payload.image_description.strip()
        if not description:
            raise HTTPException(status_code=400, detail="image_description cannot be empty")

        fact = f"Visual context captured: {description}"
        preview = memory_agent._sanitize_preview(description)
        memory_agent._append_transcript("assistant", f"[Contexte visuel enregistré : {preview}]")

        background_tasks.add_task(memory_agent._save_memory_in_background, fact, "vision")

        return StatusMessageResponse(
            status="success",
            message="Contexte visuel extrait et enregistré.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _internal_error("Failed to capture vision context", exc)


@router.post("/trigger/meeting_start", response_model=MeetingTriggerResponse)
async def trigger_meeting_start(payload: MeetingStartPayload) -> MeetingTriggerResponse:
    fallback_memory = "On lui a demandé les résultats du sondage il y a 4 heures."

    try:
        attendee_name = payload.attendee_name.strip()
        if not attendee_name:
            raise HTTPException(status_code=400, detail="attendee_name cannot be empty")

        selected_memory = fallback_memory

        try:
            client = memory_agent._get_mem0_client()
            try:
                raw_memories = await run_in_threadpool(
                    client.get_all,
                    filters={"user_id": DEMO_USER_ID},
                )
            except Exception:
                raw_memories = await run_in_threadpool(client.get_all, user_id=DEMO_USER_ID)

            normalized = memory_agent._normalize_memories(raw_memories)
            selected_memory = memory_agent._select_meeting_memory(
                attendee_name=attendee_name,
                memories=normalized,
                fallback=fallback_memory,
            )
        except Exception:
            selected_memory = fallback_memory

        whisper_generated = memory_agent._build_meeting_whisper(
            attendee_name=attendee_name,
            selected_memory=selected_memory,
        )
        memory_agent._append_transcript(
            "assistant",
            memory_agent._format_proactive_whisper_transcript(whisper_generated),
        )

        return MeetingTriggerResponse(status="success", whisper_generated=whisper_generated)
    except HTTPException:
        raise
    except Exception as exc:
        _internal_error("Failed to trigger meeting whisper", exc)


# ---------------------------------------------------------------------------
# Vapi management endpoints
# ---------------------------------------------------------------------------


@router.get("/vapi/calls", response_model=VapiCallsResponse)
async def get_vapi_calls(limit: int = Query(default=10, ge=1, le=100)) -> VapiCallsResponse:
    """Return the most-recent *limit* calls from the Vapi dashboard."""
    raw = await vapi_client.list_calls(limit=limit)
    calls = [
        VapiCallItem(
            id=c.get("id", ""),
            status=c.get("status"),
            createdAt=c.get("createdAt"),
            endedAt=c.get("endedAt"),
        )
        for c in raw
        if c.get("id")
    ]
    return VapiCallsResponse(status="success", calls=calls)


@router.post("/vapi/provision", response_model=VapiProvisionResponse)
async def provision_vapi_assistant(payload: VapiProvisionPayload) -> VapiProvisionResponse:
    """Crée (ou reconfigure) l'assistant Memento sur Vapi.

    The assistant is provisioned with:
    * Both custom tools pointing back to this backend's webhook endpoints.
    * The caller-supplied *system_prompt* as the model system message.
    * ``vapi_server_url`` from settings as the server URL for end-of-call events.
    """
    if not payload.system_prompt.strip():
        raise HTTPException(status_code=400, detail="system_prompt cannot be empty")

    name = (payload.assistant_name or "Memento").strip() or "Memento"
    data = await vapi_client.create_assistant(
        system_prompt=payload.system_prompt.strip(),
        name=name,
    )
    assistant_id = data.get("id", "")
    return VapiProvisionResponse(
        status="success",
        assistant_id=assistant_id,
        message=f"Assistant « {name} » provisionné (id={assistant_id}).",
    )


@router.post("/vapi/inject", response_model=VapiInjectResponse)
async def inject_vapi_message(payload: VapiInjectPayload) -> VapiInjectResponse:
    """Inject a spoken message into an active Vapi call.

    This route uses the live call ``controlUrl`` returned by Vapi for active
    calls. Ended calls are rejected with a clear 409 instead of failing with an
    opaque upstream 404.
    """
    call_id = payload.call_id.strip()
    message = payload.message.strip()
    if not call_id:
        raise HTTPException(status_code=400, detail="call_id cannot be empty")
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    await vapi_client.say_in_call(
        call_id=call_id,
        message=message,
        end_call_after=payload.end_call_after,
    )
    memory_agent._append_transcript(
        "assistant",
        f'[Injection Vapi → appel={call_id} : « {memory_agent._sanitize_preview(message)} »]',
    )
    return VapiInjectResponse(
        status="success",
        message=f"Message injecté dans l'appel {call_id}.",
    )


# ---------------------------------------------------------------------------
# Memento Guardian : détection visage et reconnaissance
# ---------------------------------------------------------------------------


@router.post("/vision/detect", response_model=VisionDetectResponse)
async def detect_faces_and_lookup_memory(
    payload: VisionDetectPayload,
) -> VisionDetectResponse:
    """
    Pont proactif — Memento Guardian

    1. Recoit une frame webcam (base64)
    2. Detecte les visages avec face_recognition
    3. Si un visage est identifie -> Interroge Mem0 pour cette personne
    4. Retourne les matches memoire + declenche une alerte si tache trouvee
    """
    try:
        # Etape 1: Detection faciale
        detected_faces = await run_in_threadpool(
            face_service.detect_faces_in_frame,
            payload.image_data,
        )

        if not detected_faces:
            return VisionDetectResponse(
                status="success",
                faces_detected=0,
                persons=[],
                memory_matches=[],
                proactive_alert=None,
            )

        persons: list[DetectedPerson] = []
        all_memory_matches: list[MemoryMatch] = []
        proactive_alerts: list[str] = []

        proactive_suggestion: str | None = None
        primary_identified: str | None = None
        seen_memory_ids: set[str] = set()

        for face in detected_faces:
            person = DetectedPerson(
                name=face["name"],
                confidence=face["confidence"],
                identified=face["identified"],
                bounding_box=BoundingBox(**face["bounding_box"]),
            )
            persons.append(person)

            # Etape 2 : réseau contextuel (visage + entourage proche dans Mem0)
            if face["identified"] and face["name"] != "Inconnu":
                memories = await run_in_threadpool(
                    memory_agent.search_memories_for_contextual_network,
                    face["name"],
                    translate_to_french=True,
                )

                for mem in memories:
                    mid = str(mem.get("id", ""))
                    if mid and mid in seen_memory_ids:
                        continue
                    if mid:
                        seen_memory_ids.add(mid)

                    match = MemoryMatch(
                        text=mem["text"],
                        category=mem["category"],
                        relevance=mem["relevance"],
                    )
                    all_memory_matches.append(match)

                    category_lower = mem["category"].lower()
                    if category_lower in ("task", "task_for_person", "commitment", "voice"):
                        proactive_alerts.append(
                            f"Rappel pour {face['name']}: {mem['text']}"
                        )

                if primary_identified is None and memories:
                    primary_identified = face["name"]
                    fact_texts = [m["text"] for m in memories if m.get("text")]
                    proactive_suggestion = await run_in_threadpool(
                        memory_agent.generate_proactive_conversation_suggestion,
                        face["name"],
                        fact_texts,
                        settings.memento_viewer_name,
                    )

        # Etape 3 : alerte proactive — phrase LLM prioritaire, sinon rappels tâches
        proactive_alert = None
        if proactive_suggestion:
            proactive_alert = proactive_suggestion
        elif proactive_alerts:
            proactive_alert = " | ".join(proactive_alerts[:3])

        if proactive_alert:
            memory_agent._append_transcript(
                "assistant",
                f"[MEMENTO GUARDIAN PROACTIF : visage reconnu → {proactive_alert}]",
            )

        return VisionDetectResponse(
            status="success",
            faces_detected=len(detected_faces),
            persons=persons,
            memory_matches=all_memory_matches,
            proactive_alert=proactive_alert,
            proactive_suggestion=proactive_suggestion,
        )

    except Exception as exc:
        logger.exception("Vision detection failed", exc_info=exc)
        raise HTTPException(status_code=500, detail="Vision detection failed")


@router.post("/faces/register", response_model=StatusMessageResponse)
async def register_new_face(payload: FaceRegisterPayload) -> StatusMessageResponse:
    """Enregistre un nouveau visage dans la base de connaissances."""
    try:
        success = await run_in_threadpool(
            face_service.add_known_face,
            payload.name,
            payload.image_data,
        )

        if success:
            return StatusMessageResponse(
                status="success",
                message=f"Visage de {payload.name} enregistre avec succes.",
            )

        raise HTTPException(
            status_code=400,
            detail="Impossible d'enregistrer le visage. Aucun visage detecte dans l'image.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Face registration failed", exc_info=exc)
        raise HTTPException(status_code=500, detail="Face registration failed")


@router.delete("/faces/{name}", response_model=StatusMessageResponse)
async def remove_face(name: str) -> StatusMessageResponse:
    """Supprime un visage de la base de connaissances."""
    success = await run_in_threadpool(face_service.remove_known_face, name)

    if success:
        return StatusMessageResponse(
            status="success",
            message=f"Visage de {name} supprime.",
        )

    raise HTTPException(status_code=404, detail=f"Visage '{name}' non trouve.")


@router.get("/faces/known", response_model=KnownFacesResponse)
async def list_known_faces() -> KnownFacesResponse:
    """Liste les personnes enregistrees pour la reconnaissance faciale."""
    known = face_service.get_known_faces_list()

    return KnownFacesResponse(
        status="success",
        known_persons=known,
    )


@router.post("/faces/reload", response_model=StatusMessageResponse)
async def reload_faces() -> StatusMessageResponse:
    """Recharge les visages depuis le dossier known_faces/."""
    count = await run_in_threadpool(face_service.reload_known_faces)

    return StatusMessageResponse(
        status="success",
        message=f"{count} visage(s) charge(s) depuis le dossier.",
    )


@router.get("/vision/status")
async def vision_status() -> dict:
    """Verifie si le service de reconnaissance faciale est disponible."""
    available = face_service.is_available()
    known_count = len(face_service.get_known_faces_list())

    return {
        "status": "success",
        "face_recognition_available": available,
        "known_faces_count": known_count,
    }

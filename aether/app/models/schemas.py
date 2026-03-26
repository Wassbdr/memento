from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, RootModel


class TranscriptMessage(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    text: str


class TranscriptResponse(BaseModel):
    status: str
    messages: list[TranscriptMessage]


class MemoryItem(BaseModel):
    id: str
    text: str
    category: str
    timestamp: str


class MemoriesResponse(BaseModel):
    status: str
    data: list[MemoryItem]


class ContextResponse(BaseModel):
    status: str
    injected_prompt: str


class ToolModel(BaseModel):
    name: str | None = None


class ToolCallArguments(BaseModel):
    fact_to_remember: str | None = None
    category: str | None = None


class ToolCallModel(BaseModel):
    id: str | None = None
    arguments: ToolCallArguments | dict[str, Any] | str | None = None


class ToolWithToolCallModel(BaseModel):
    tool: ToolModel | None = None
    toolCall: ToolCallModel | None = None


class WebhookMessageModel(BaseModel):
    type: str | None = None
    toolWithToolCallList: list[ToolWithToolCallModel] = Field(default_factory=list)


class WebhookPayload(BaseModel):
    message: WebhookMessageModel | None = None


class ToolResultModel(BaseModel):
    toolCallId: str
    result: str


class WebhookResponse(BaseModel):
    results: list[ToolResultModel]


class CallEndedPayload(RootModel[dict[str, Any]]):
    pass


class CallReportItem(BaseModel):
    timestamp: str
    summary: str
    transcript_excerpt: str | None = None


class CallEndedResponse(BaseModel):
    status: str
    report: str


class ReportsResponse(BaseModel):
    status: str
    reports: list[CallReportItem]


class StatusMessageResponse(BaseModel):
    status: str
    message: str


class VisionCapturePayload(BaseModel):
    image_description: str = Field(min_length=1, max_length=2000)


class MeetingStartPayload(BaseModel):
    attendee_name: str = Field(min_length=1, max_length=120)


class MeetingTriggerResponse(BaseModel):
    status: str
    whisper_generated: str


# ---------------------------------------------------------------------------
# Vapi
# ---------------------------------------------------------------------------


class VapiCallItem(BaseModel):
    id: str
    status: str | None = None
    createdAt: str | None = None
    endedAt: str | None = None


class VapiCallsResponse(BaseModel):
    status: str
    calls: list[VapiCallItem]


class VapiProvisionPayload(BaseModel):
    system_prompt: str = Field(min_length=1, max_length=8000)
    assistant_name: str | None = "Memento"


class VapiProvisionResponse(BaseModel):
    status: str
    assistant_id: str
    message: str


class VapiInjectPayload(BaseModel):
    call_id: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=2000)
    end_call_after: bool = False


class VapiInjectResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Aether Guardian: Vision Detection
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    top: int
    right: int
    bottom: int
    left: int


class DetectedPerson(BaseModel):
    name: str
    confidence: float
    identified: bool
    bounding_box: BoundingBox


class MemoryMatch(BaseModel):
    text: str
    category: str
    relevance: str


class VisionDetectPayload(BaseModel):
    image_data: str = Field(min_length=100, description="Image en base64 (data URL ou raw)")


class VisionDetectResponse(BaseModel):
    status: str
    faces_detected: int
    persons: list[DetectedPerson]
    memory_matches: list[MemoryMatch]
    proactive_alert: str | None = None
    # Phrase générée (OpenRouter) pour engager la conversation — prioritaire pour l’UI.
    proactive_suggestion: str | None = None


class FaceRegisterPayload(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    image_data: str = Field(min_length=100, description="Image en base64")


class KnownFacesResponse(BaseModel):
    status: str
    known_persons: list[str]


class MemoryClassification(BaseModel):
    type: str  # fact, preference, task_for_person, question, unknown
    extracted_entity: str | None = None
    action: str | None = None
    content: str | None = None
    confidence: float = 0.0

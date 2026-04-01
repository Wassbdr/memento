"""Project planning helpers derived from the README."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkItem:
    title: str
    objective: str
    deliverable: str
    completed: bool = False


@dataclass(frozen=True)
class Workstream:
    name: str
    description: str
    items: tuple[WorkItem, ...]


@dataclass(frozen=True)
class ProjectScope:
    product_vision: str
    workstreams: tuple[Workstream, ...]

    def to_markdown(self) -> str:
        lines = [
            "# TODO",
            "",
            "## Vision",
            self.product_vision,
            "",
            "## Chantiers",
            "",
        ]
        for workstream in self.workstreams:
            lines.append(f"### {workstream.name}")
            lines.append(workstream.description)
            lines.append("")
            for item in workstream.items:
                marker = "x" if item.completed else " "
                lines.append(f"- [{marker}] {item.title}")
                lines.append(f"  Objectif : {_decapitalize(item.objective)}")
                lines.append(f"  Livrable : {_decapitalize(item.deliverable)}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def build_default_scope() -> ProjectScope:
    return ProjectScope(
        product_vision=(
            "Construire une prothese cognitive vocale rassurante, sans ecran, "
            "capable de capter le contexte du quotidien, de le structurer dans "
            "une memoire personnalisee et de repondre naturellement au patient."
        ),
        workstreams=(
            Workstream(
                name="1. Capture audio",
                description=(
                    "Stabiliser l'entree micro et l'activation vocale pour "
                    "obtenir des interactions fiables et non intrusives."
                ),
                items=(
                    WorkItem(
                        title="Integrer le micro temps reel",
                        objective="Capturer un flux audio continu exploitable localement.",
                        deliverable="Un module d'acquisition audio avec configuration du device et logs de sante.",
                        completed=True,
                    ),
                    WorkItem(
                        title="Ajouter la detection de parole (VAD)",
                        objective="Ne declencher les traitements que pendant les segments utiles.",
                        deliverable="Une brique VAD testee sur silence, parole et bruit de fond.",
                        completed=True,
                    ),
                    WorkItem(
                        title="Brancher la transcription Whisper",
                        objective="Transformer les tours de parole en texte de maniere robuste.",
                        deliverable="Un pipeline STT documente avec mesure de latence et de qualite.",
                        completed=True,
                    ),
                ),
            ),
            Workstream(
                name="2. Memoire et contexte",
                description=(
                    "Representer la memoire personnelle du patient et relier "
                    "les evenements, personnes, lieux et emotions."
                ),
                items=(
                    WorkItem(
                        title="Modeliser le graphe memoire",
                        objective="Definir les entites, relations et attributs affectifs du domaine.",
                        deliverable="Un schema initial pour Neo4j avec exemples de noeuds et relations.",
                    ),
                    WorkItem(
                        title="Mettre en place l'indexation semantique",
                        objective="Permettre la recherche contextuelle dans les souvenirs et habitudes.",
                        deliverable="Une couche LlamaIndex connectee a ChromaDB pour l'ingestion et la recherche.",
                    ),
                    WorkItem(
                        title="Concevoir la strategie de synchronisation",
                        objective="Faire coexister recherche vectorielle et graphe de connaissances.",
                        deliverable="Des regles d'ecriture/lecture entre ChromaDB et Neo4j avec tests d'integration.",
                    ),
                ),
            ),
            Workstream(
                name="3. Raisonnement conversationnel",
                description=(
                    "Produire des reponses utiles, apaisantes et contextualisees "
                    "a partir de la memoire du patient."
                ),
                items=(
                    WorkItem(
                        title="Definir la boucle RAG conversationnelle",
                        objective="Assembler recuperation de contexte, prompt et generation de reponse.",
                        deliverable="Un orchestrateur applicatif branche sur Ministral 3 8B.",
                    ),
                    WorkItem(
                        title="Formuler des garde-fous conversationnels",
                        objective="Limiter les hallucinations et adopter un ton rassurant adapte au public cible.",
                        deliverable="Un jeu de prompts systeme et de cas limites documentes.",
                    ),
                    WorkItem(
                        title="Tracer les decisions du systeme",
                        objective="Pouvoir expliquer quelle memoire a ete mobilisee pour chaque reponse.",
                        deliverable="Des logs ou traces de retrieval exploitables pour debug et evaluation.",
                    ),
                ),
            ),
            Workstream(
                name="4. Restitution vocale",
                description=(
                    "Transformer les reponses texte en experience vocale fluide, "
                    "claire et rassurante."
                ),
                items=(
                    WorkItem(
                        title="Connecter Voxtral TTS",
                        objective="Generer une voix naturelle et intelligible.",
                        deliverable="Un adaptateur TTS configurable avec choix de voix et parametres de synthese.",
                    ),
                    WorkItem(
                        title="Diffuser la reponse sur haut-parleur",
                        objective="Assurer une sortie audio stable et a faible latence.",
                        deliverable="Un module de playback avec gestion d'erreurs et interruptions.",
                    ),
                    WorkItem(
                        title="Mesurer l'experience temps reel",
                        objective="Verifier que le cycle complet reste supportable pour un usage quotidien.",
                        deliverable="Des indicateurs de latence bout-en-bout et seuils cibles.",
                    ),
                ),
            ),
            Workstream(
                name="5. Produit et confiance",
                description=(
                    "Encadrer les usages, le consentement et la qualite globale "
                    "du systeme avant d'aller vers des interfaces plus riches."
                ),
                items=(
                    WorkItem(
                        title="Definir l'apport des aidants et de la famille",
                        objective="Structurer la collecte manuelle de souvenirs, routines et profils proches.",
                        deliverable="Un backlog fonctionnel de l'application compagnon et de ses flux de donnees.",
                    ),
                    WorkItem(
                        title="Cadencer la collecte contextuelle consentie",
                        objective="Encadrer les captures automatiques de conversations, visites et emotions.",
                        deliverable="Une politique de consentement, retention et suppression des donnees.",
                    ),
                    WorkItem(
                        title="Preparer l'extension visuelle future",
                        objective="Garder une architecture compatible avec lunettes connectees et reconnaissance faciale.",
                        deliverable="Une note d'architecture listant les points d'extension et les risques.",
                    ),
                ),
            ),
        ),
    )


def _decapitalize(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]

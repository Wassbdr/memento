const CATEGORY_LABEL_FR = {
  task: "Tâche",
  task_for_person: "Tâche (personne)",
  voice: "Voix",
  family: "Famille",
  meeting: "Réunion",
  commitment: "Engagement",
  personal: "Personnel",
  work: "Travail",
  general: "Général",
};

/**
 * RealTimeHelpPanel — Memento Guardian
 *
 * Panneau d'aide en temps réel : personnes détectées et mémoires associées.
 */
export default function RealTimeHelpPanel({ detectedPersons, memoryMatches }) {
  const hasDetections = detectedPersons && detectedPersons.length > 0;
  const hasMemories = memoryMatches && memoryMatches.length > 0;

  return (
    <div
      className="rounded-2xl p-4 overflow-auto h-full"
      style={{
        background: "rgba(0,0,0,0.2)",
        border: "1px solid rgba(255,255,255,0.1)",
      }}
    >
      <h3 className="text-sm font-semibold text-white/90 mb-3 flex items-center gap-2">
        <span>Aide en temps réel</span>
        {hasDetections && (
          <span className="px-2 py-0.5 text-xs bg-emerald-500/30 text-emerald-300 rounded-full">
            {detectedPersons.length} personne(s)
          </span>
        )}
      </h3>

      {/* Personnes detectees */}
      {hasDetections ? (
        <div className="space-y-2 mb-4">
          {detectedPersons.map((person, idx) => (
            <div
              key={idx}
              className="p-3 rounded-lg bg-white/5 border border-white/10"
            >
              <div className="flex items-center gap-2">
                <span className="text-lg">
                  {person.identified ? "✅" : "❓"}
                </span>
                <span className="font-medium text-white/90">{person.name}</span>
                {person.identified && (
                  <span className="text-xs text-emerald-400 ml-auto">
                    {Math.round(person.confidence * 100)}% confiance
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-white/50 text-sm mb-4">
          Aucune personne detectee. Activez la webcam pour commencer.
        </p>
      )}

      {/* Memoires associees */}
      {hasMemories && (
        <>
          <h4 className="text-xs font-semibold text-white/70 mb-2 uppercase tracking-wider">
            Mémoires associées
          </h4>
          <div className="space-y-2">
            {memoryMatches.slice(0, 5).map((memory, idx) => (
              <div
                key={idx}
                className="p-2 rounded-lg text-sm"
                style={{
                  background:
                    memory.category === "task" || memory.category === "task_for_person"
                      ? "rgba(251,191,36,0.15)"
                      : "rgba(99,102,241,0.15)",
                  border:
                    memory.category === "task" || memory.category === "task_for_person"
                      ? "1px solid rgba(251,191,36,0.3)"
                      : "1px solid rgba(99,102,241,0.2)",
                }}
              >
                <p className="text-white/90">{memory.text}</p>
                <div className="flex items-center gap-2 mt-1">
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      memory.category === "task" || memory.category === "task_for_person"
                        ? "bg-amber-500/20 text-amber-300"
                        : "bg-indigo-500/20 text-indigo-300"
                    }`}
                  >
                    {CATEGORY_LABEL_FR[memory.category] ?? memory.category}
                  </span>
                  <span className="text-xs text-white/40">{memory.relevance}</span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Empty state avec instructions */}
      {!hasDetections && !hasMemories && (
        <div className="mt-4 p-3 rounded-lg bg-white/5 border border-dashed border-white/20">
          <p className="text-xs text-white/50 leading-relaxed">
            <strong className="text-white/70">Comment ça marche :</strong>
            <br />
            1. Démarrez la webcam ci-dessus
            <br />
            2. Les visages connus seront identifiés
            <br />
            3. Les rappels associés s’affichent ici
          </p>
        </div>
      )}
    </div>
  );
}

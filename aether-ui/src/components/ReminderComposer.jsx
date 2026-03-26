import { useState } from "react";
import { Loader2, NotebookPen, Send, Tag } from "lucide-react";

const CATEGORIES = [
  "personal",
  "work",
  "meeting",
  "preference",
  "contact",
  "task",
  "general",
];

const CATEGORY_LABELS = {
  personal: "Personnel",
  work: "Travail",
  meeting: "Réunion",
  preference: "Préférence",
  contact: "Contact",
  task: "Tâche",
  general: "Général",
};

export default function ReminderComposer({ onSaveReminder, disabled }) {
  const [text, setText] = useState("");
  const [category, setCategory] = useState("personal");
  const [saving, setSaving] = useState(false);

  const canSubmit = text.trim().length > 0 && !saving && !disabled;

  const onSubmit = async (event) => {
    event.preventDefault();
    const value = text.trim();
    if (!value) return;

    setSaving(true);
    try {
      await onSaveReminder(value, category);
      setText("");
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="shrink-0 rounded-xl border border-zinc-800/60 bg-zinc-900/40 backdrop-blur-sm p-4">
      <div className="flex items-center gap-2.5 mb-3">
        <NotebookPen className="w-4 h-4 text-indigo-400" />
        <h2 className="text-sm font-semibold text-zinc-200">Demander à Memento de retenir</h2>
      </div>

      <form onSubmit={onSubmit} className="flex flex-col md:flex-row gap-2.5">
        <div className="relative flex-1">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Ex. : Rappelle-moi d’envoyer le dossier à Sarah avant 16 h"
            className="w-full rounded-lg border border-zinc-700/80 bg-zinc-900/70 px-3.5 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 outline-none focus:border-indigo-500/70 focus:ring-2 focus:ring-indigo-500/20 disabled:opacity-60"
            disabled={saving || disabled}
            maxLength={220}
          />
        </div>

        <label className="relative inline-flex items-center">
          <Tag className="absolute left-2.5 w-3.5 h-3.5 text-zinc-500 pointer-events-none" />
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="appearance-none rounded-lg border border-zinc-700/80 bg-zinc-900/70 pl-8 pr-8 py-2.5 text-xs uppercase tracking-wider text-zinc-300 outline-none focus:border-indigo-500/70 focus:ring-2 focus:ring-indigo-500/20 disabled:opacity-60"
            disabled={saving || disabled}
          >
            {CATEGORIES.map((item) => (
              <option key={item} value={item}>
                {CATEGORY_LABELS[item] ?? item}
              </option>
            ))}
          </select>
        </label>

        <button
          type="submit"
          disabled={!canSubmit}
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-indigo-500/40 bg-indigo-600/20 px-4 py-2.5 text-xs font-semibold tracking-wide text-indigo-200 transition hover:bg-indigo-600/30 hover:border-indigo-400/60 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {saving ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Enregistrement…
            </>
          ) : (
            <>
              <Send className="w-3.5 h-3.5" />
              Enregistrer
            </>
          )}
        </button>
      </form>

      <p className="mt-2 text-[11px] text-zinc-500">
        Les rappels apparaissent dans le graphe et nourrissent le contexte proactif.
      </p>
    </section>
  );
}

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  Clock,
  Heart,
  Activity,
  User,
  ChevronDown,
} from "lucide-react";
import { MOCK_ALERTS, type Alert, type AlertLevel } from "../data/mockData";

// ─── Level Configuration ──────────────────────────────────────────────────────

type LevelConfig = {
  label: string;
  color: string;
  bg: string;
  border: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  Icon: React.FC<any>;
};

const LEVEL_CONFIG: Record<AlertLevel, LevelConfig> = {
  clinique: {
    label: "Clinique",
    color: "text-amber-200",
    bg: "bg-amber-600/20",
    border: "border-amber-500/40",
    Icon: AlertTriangle,
  },
  routine: {
    label: "Routine",
    color: "text-sky-200",
    bg: "bg-sky-600/20",
    border: "border-sky-500/40",
    Icon: Activity,
  },
  emotionnel: {
    label: "Émotionnel",
    color: "text-fuchsia-200",
    bg: "bg-fuchsia-600/20",
    border: "border-fuchsia-500/40",
    Icon: Heart,
  },
};

const LEVEL_ORDER: AlertLevel[] = ["clinique", "emotionnel", "routine"];

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatTime(ts: string): string {
  return new Date(ts).toLocaleTimeString("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDate(ts: string): string {
  const d = new Date(ts);
  const today = new Date();
  if (d.toDateString() === today.toDateString()) return "Aujourd'hui";
  return d.toLocaleDateString("fr-FR", { day: "numeric", month: "short" });
}

// ─── Alert Card ───────────────────────────────────────────────────────────────

const cardVariants = {
  hidden: { opacity: 0, y: 18, scale: 0.97 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      delay: i * 0.07,
      duration: 0.38,
      ease: [0.25, 0.1, 0.25, 1],
    },
  }),
  exit: {
    opacity: 0,
    scale: 0.96,
    y: -8,
    transition: { duration: 0.2 },
  },
};

function AlertCard({ alert, index }: { alert: Alert; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const cfg = LEVEL_CONFIG[alert.level];
  const { Icon } = cfg;

  return (
    <motion.article
      custom={index}
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      exit="exit"
      layout
      className={`glass rounded-[var(--radius-lg)] p-4 cursor-pointer select-none border ${cfg.border} transition-colors duration-200 hover:bg-white/[0.08]`}
      onClick={() => setExpanded((v) => !v)}
    >
      <div className="flex items-start gap-3">
        {/* Level icon */}
        <div
          className={`mt-0.5 p-2 rounded-xl ${cfg.bg} flex-shrink-0 border ${cfg.border}`}
        >
          <Icon className={`w-3.5 h-3.5 ${cfg.color}`} />
        </div>

        <div className="flex-1 min-w-0">
          {/* Title + badge + chevron */}
          <div className="flex items-start justify-between gap-2 mb-1.5">
            <h3 className="text-sm font-semibold text-white/95 leading-snug">
              {alert.title}
            </h3>
            <div className="flex items-center gap-1.5 flex-shrink-0 mt-0.5">
              <span
                className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full ${cfg.bg} ${cfg.color} border ${cfg.border}`}
              >
                {cfg.label}
              </span>
              <motion.span
                animate={{ rotate: expanded ? 180 : 0 }}
                transition={{ duration: 0.2 }}
                className="block"
              >
                <ChevronDown className="w-3.5 h-3.5 text-white/40" />
              </motion.span>
            </div>
          </div>

          {/* Meta — time & person */}
          <div className="flex items-center gap-3 text-[11px] text-white/55 mb-2">
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {formatDate(alert.timestamp)} · {formatTime(alert.timestamp)}
            </span>
            {alert.relatedPerson && (
              <span className="flex items-center gap-1">
                <User className="w-3 h-3" />
                {alert.relatedPerson}
              </span>
            )}
          </div>

          {/* Collapsed preview */}
          {!expanded && (
            <p className="text-xs text-white/65 line-clamp-1">
              {alert.description}
            </p>
          )}

          {/* Expanded body */}
          <AnimatePresence initial={false}>
            {expanded && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.22, ease: "easeInOut" }}
                className="overflow-hidden"
              >
                <p className="text-xs text-white/75 leading-relaxed mb-3">
                  {alert.description}
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {alert.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-[10px] px-2 py-0.5 rounded-[var(--radius-full)] bg-white/8 text-white/60 border border-white/15"
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.article>
  );
}

// ─── Alert Feed ───────────────────────────────────────────────────────────────

type Filter = AlertLevel | "all";

export default function AlertFeed() {
  const [filter, setFilter] = useState<Filter>("all");

  const sorted = [...MOCK_ALERTS].sort((a, b) => {
    const lvl =
      LEVEL_ORDER.indexOf(a.level) - LEVEL_ORDER.indexOf(b.level);
    if (lvl !== 0) return lvl;
    return (
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  });

  const visible = filter === "all" ? sorted : sorted.filter((a) => a.level === filter);

  const counts: Record<Filter, number> = {
    all: MOCK_ALERTS.length,
    clinique: MOCK_ALERTS.filter((a) => a.level === "clinique").length,
    emotionnel: MOCK_ALERTS.filter((a) => a.level === "emotionnel").length,
    routine: MOCK_ALERTS.filter((a) => a.level === "routine").length,
  };

  return (
    <div className="h-full flex flex-col px-5 sm:px-7 lg:px-10 py-8 max-w-3xl mx-auto w-full section-compact">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
        className="mb-7 flex-shrink-0"
      >
        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-subtle mb-2">
          Fil de suivi
        </p>
        <h1 className="text-section text-white leading-tight">
          Préoccupations &amp; Soins
        </h1>
        <p className="text-sm text-muted mt-1">Mamie Rose · Aujourd'hui</p>
      </motion.header>

      {/* Filter chips */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.12 }}
        className="flex gap-2 mb-5 flex-wrap flex-shrink-0"
      >
        {(["all", "clinique", "emotionnel", "routine"] as Filter[]).map(
          (f) => {
            const active = filter === f;
            const cfg = f !== "all" ? LEVEL_CONFIG[f] : null;
            return (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`cursor-pointer px-3 py-1.5 rounded-[var(--radius-full)] text-xs font-medium transition-colors duration-200 border ${
                  active
                    ? cfg
                      ? `${cfg.bg} ${cfg.color} ${cfg.border}`
                      : "bg-white/14 text-white border-white/30"
                    : "bg-transparent text-white/55 border-white/15 hover:text-white/80 hover:border-white/28"
                }`}
              >
                {f === "all" ? "Tout" : LEVEL_CONFIG[f].label}
                <span className="ml-1.5 opacity-50">({counts[f]})</span>
              </button>
            );
          }
        )}
      </motion.div>

      {/* Cards list */}
      <div className="flex-1 overflow-y-auto space-y-2.5 pr-1 min-h-0">
        <AnimatePresence mode="popLayout">
          {visible.map((alert, i) => (
            <AlertCard key={alert.id} alert={alert} index={i} />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}

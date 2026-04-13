import { useState } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import {
  Sparkles,
  Activity,
  Brain,
  Mic,
  MicOff,
  ArrowRight,
  type LucideProps,
} from "lucide-react";
import OrganicSphere, { type SphereState } from "./components/OrganicSphere";
import AlertFeed from "./components/AlertFeed";
import MemoryGraph from "./components/MemoryGraph";

// ─── Navigation ───────────────────────────────────────────────────────────────

type Panel = "soul" | "feed" | "brain";

const NAV: { id: Panel; label: string; Icon: React.FC<LucideProps> }[] = [
  { id: "soul", label: "L'Âme", Icon: Sparkles },
  { id: "feed", label: "Fil de Soin", Icon: Activity },
  { id: "brain", label: "Le Cerveau", Icon: Brain },
];

const PANEL_ORDER: Panel[] = ["soul", "feed", "brain"];

// ─── Panel transition variants ────────────────────────────────────────────────

const panelVariants = {
  enter: (dir: number) => ({
    opacity: 0,
    x: dir > 0 ? 40 : -40,
  }),
  center: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] },
  },
  exit: (dir: number) => ({
    opacity: 0,
    x: dir > 0 ? -40 : 40,
    transition: { duration: 0.24, ease: [0.22, 1, 0.36, 1] },
  }),
};

const heroContainer = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.12,
    },
  },
};

const heroItem = {
  hidden: { opacity: 0, y: 22 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.55,
      ease: [0.22, 1, 0.36, 1],
    },
  },
};

// ─── Soul Panel ───────────────────────────────────────────────────────────────

function SoulPanel({
  sphereState,
  onToggle,
  onOpenFeed,
  onOpenBrain,
  micError,
}: {
  sphereState: SphereState;
  onToggle: () => void;
  onOpenFeed: () => void;
  onOpenBrain: () => void;
  micError: string | null;
}) {
  const listening = sphereState === "listening";

  return (
    <section className="w-full h-full overflow-y-auto px-4 sm:px-6 lg:px-10 py-4 sm:py-6 lg:py-8">
      <div className="mx-auto min-h-full w-full max-w-[1280px] grid grid-cols-1 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)] gap-6 lg:gap-12 items-center">
        <motion.div
          variants={heroContainer}
          initial="hidden"
          animate="visible"
          className="order-1 flex flex-col gap-6 lg:gap-8"
        >
          <motion.p
            variants={heroItem}
            className="text-[11px] uppercase tracking-[0.32em] text-subtle"
          >
            Memento · Interface Aidant
          </motion.p>

          <motion.h1 variants={heroItem} className="text-hero max-w-[16ch]">
            Rester proche, meme quand les souvenirs se brouillent.
          </motion.h1>

          <motion.p variants={heroItem} className="text-subhead text-muted max-w-[46ch]">
            Une interface claire pour capter les signaux faibles, structurer le
            suivi et garder un lien humain au quotidien.
          </motion.p>

          <motion.div variants={heroItem} className="flex flex-wrap gap-3.5">
            <button
              onClick={onToggle}
              className="cursor-pointer inline-flex items-center gap-2.5 rounded-[var(--radius-md)] bg-white text-black px-6 py-3 text-sm font-semibold transition-colors duration-200 hover:bg-white/90"
            >
              {listening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
              {listening ? "Arrêter l'écoute" : "Parler à Memento"}
            </button>

            <button
              onClick={onOpenFeed}
              className="cursor-pointer group inline-flex items-center gap-2.5 rounded-[var(--radius-md)] border border-white/30 px-6 py-3 text-sm font-medium text-white transition-colors duration-200 hover:bg-white/10"
            >
              Voir le fil de soin
              <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-0.5" />
            </button>
          </motion.div>

          {micError && (
            <motion.p
              variants={heroItem}
              className="text-xs sm:text-sm text-rose-200/95 border border-rose-300/30 bg-rose-400/12 rounded-[var(--radius-md)] px-3 py-2 w-fit"
            >
              {micError}
            </motion.p>
          )}

          <motion.div variants={heroItem} className="flex flex-wrap items-center gap-2.5 text-xs text-subtle">
            <button
              onClick={onOpenBrain}
              className="cursor-pointer rounded-[var(--radius-full)] border border-white/16 px-3 py-1.5 transition-colors duration-200 hover:bg-white/8"
            >
              Explorer le cerveau mémoire
            </button>
            <span className="rounded-[var(--radius-full)] border border-white/12 px-3 py-1.5">
              {listening ? "Micro actif" : "Micro en attente"}
            </span>
          </motion.div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.18, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="order-2 relative h-[280px] sm:h-[340px] lg:h-[min(68vh,620px)] min-h-[280px] ui-panel overflow-hidden"
        >
          <div className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-center justify-between border-b border-white/10 px-4 py-3 sm:px-6">
            <div>
              <p className="text-[10px] uppercase tracking-[0.3em] text-white/35">
                Memento · Profil
              </p>
              <p className="text-sm font-display font-medium text-white/80">Mamie Rose</p>
            </div>
            <span className="rounded-[var(--radius-full)] border border-white/15 bg-white/5 px-2.5 py-1 text-[10px] uppercase tracking-[0.18em] text-white/55">
              {listening ? "Listening" : "Idle"}
            </span>
          </div>

          <div className="absolute inset-0 pt-14">
            <OrganicSphere state={sphereState} />
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [panel, setPanel] = useState<Panel>("soul");
  const [dir, setDir] = useState(1);
  const [sphereState, setSphereState] = useState<SphereState>("idle");
  const [micError, setMicError] = useState<string | null>(null);
  const [hasMicPermission, setHasMicPermission] = useState<boolean | null>(null);
  const reduceMotion = useReducedMotion();

  const navigate = (next: Panel) => {
    const ci = PANEL_ORDER.indexOf(panel);
    const ni = PANEL_ORDER.indexOf(next);
    setDir(ni >= ci ? 1 : -1);
    setPanel(next);
  };

  const ensureMicPermission = async (): Promise<boolean> => {
    if (hasMicPermission === true) return true;
    if (!navigator.mediaDevices?.getUserMedia) {
      setMicError("Le micro n'est pas supporté sur ce navigateur.");
      setHasMicPermission(false);
      return false;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
      setHasMicPermission(true);
      setMicError(null);
      return true;
    } catch {
      setHasMicPermission(false);
      setMicError("Accès micro refusé. Autorise le micro dans le navigateur puis réessaie.");
      return false;
    }
  };

  const toggleListening = async () => {
    if (sphereState === "listening") {
      setSphereState("idle");
      setMicError(null);
      return;
    }

    const canListen = await ensureMicPermission();
    if (!canListen) return;
    setSphereState("listening");
  };

  return (
    <div className="w-screen h-screen bg-[var(--color-bg)] overflow-hidden flex relative">
      {/* ── Sidebar ── */}
      <aside className="relative z-20 flex flex-col items-center py-8 px-3 w-[74px] flex-shrink-0 border-r border-white/10 bg-[var(--color-bg-muted)]">
        {/* Logo mark */}
        <div className="mb-10 w-10 h-10 rounded-[var(--radius-md)] border border-white/20 bg-white/5 flex items-center justify-center">
          <span className="text-white text-xs font-display font-bold tracking-tight">
            M
          </span>
        </div>

        {/* Nav items */}
        <nav className="flex flex-col gap-1.5 flex-1">
          {NAV.map(({ id, label, Icon }) => {
            const active = panel === id;
            return (
              <div key={id} className="relative group">
                <button
                  onClick={() => navigate(id)}
                  className={`cursor-pointer relative w-11 h-11 rounded-[var(--radius-md)] flex items-center justify-center transition-colors duration-200 ${
                    active
                      ? "text-[var(--color-accent-strong)]"
                      : "text-white/35 hover:text-white/70"
                  }`}
                >
                  {active && (
                    <motion.span
                      layoutId="nav-bg"
                      className="absolute inset-0 rounded-[var(--radius-md)] bg-white/8 border border-white/16"
                      transition={{ type: "spring", stiffness: 320, damping: 30 }}
                    />
                  )}
                  <Icon
                    className="w-4 h-4 relative z-10"
                    strokeWidth={active ? 2 : 1.5}
                  />
                </button>

                {/* Tooltip */}
                <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 pointer-events-none z-50 opacity-0 group-hover:opacity-100 transition-opacity duration-150">
                  <div className="ui-panel text-white/75 text-xs px-2.5 py-1.5 whitespace-nowrap">
                    {label}
                  </div>
                </div>
              </div>
            );
          })}
        </nav>

        {/* Live indicator */}
        <div className="mt-auto flex flex-col items-center gap-1.5">
          <motion.div
            className="w-1.5 h-1.5 rounded-full bg-[var(--color-live)]"
            animate={reduceMotion ? { opacity: 0.9 } : { opacity: [0.55, 1, 0.55], scale: [1, 1.1, 1] }}
            transition={reduceMotion ? undefined : { duration: 2.6, ease: "easeInOut", repeat: Infinity }}
          />
          <span
            className="text-[8px] text-white/22 uppercase tracking-widest"
            style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
          >
            Live
          </span>
        </div>
      </aside>

      {/* ── Main content ── */}
        <main className="flex-1 relative overflow-hidden">
        <AnimatePresence custom={dir} mode="wait">
          <motion.div
            key={panel}
            custom={dir}
            variants={panelVariants}
            initial="enter"
            animate="center"
            exit="exit"
            className="absolute inset-0"
          >
            {panel === "soul" && (
              <SoulPanel
                sphereState={sphereState}
                onToggle={toggleListening}
                onOpenFeed={() => navigate("feed")}
                onOpenBrain={() => navigate("brain")}
                micError={micError}
              />
            )}
            {panel === "feed" && <AlertFeed />}
            {panel === "brain" && <MemoryGraph />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

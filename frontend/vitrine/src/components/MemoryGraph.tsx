import { useRef, useCallback, useEffect, useMemo, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { motion, AnimatePresence } from "framer-motion";
import { X, Tag, Calendar, Heart, Star } from "lucide-react";
import { MOCK_GRAPH, type GraphNode } from "../data/mockData";

// ─── Node Detail Panel ────────────────────────────────────────────────────────

const TYPE_LABELS: Record<GraphNode["type"], string> = {
  patient: "Patiente",
  person: "Personne",
  place: "Lieu",
  routine: "Routine",
  episode: "Épisode",
};

function NodeDetailPanel({
  node,
  onClose,
}: {
  node: GraphNode;
  onClose: () => void;
}) {
  return (
    <motion.aside
      initial={{ x: "100%", opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: "100%", opacity: 0 }}
      transition={{ type: "spring", stiffness: 320, damping: 32 }}
      className="absolute right-0 top-0 h-full w-72 z-30 glass border-l border-white/15 p-5 overflow-y-auto flex flex-col"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-5">
        <div className="flex-1 min-w-0 pr-2">
          <span
            className="text-[10px] uppercase tracking-widest font-semibold mb-1.5 block"
            style={{ color: node.color }}
          >
            {TYPE_LABELS[node.type]}
          </span>
          <h2 className="text-base font-display font-semibold text-white leading-snug">
            {node.name}
          </h2>
          {node.details?.subtitle && (
            <p className="text-[11px] text-white/55 mt-0.5">
              {node.details.subtitle}
            </p>
          )}
        </div>
        <button
          onClick={onClose}
          className="cursor-pointer p-1.5 rounded-[var(--radius-sm)] hover:bg-white/10 text-white/45 hover:text-white/80 transition-colors flex-shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className="space-y-3.5 flex-1">
        {node.details?.notes && (
          <div className="glass rounded-[var(--radius-md)] p-3 border border-white/12">
            <p className="text-xs text-white/72 leading-relaxed">
              {node.details.notes}
            </p>
          </div>
        )}

        {node.details?.date && (
          <div className="flex items-center gap-2 text-xs text-white/62">
            <Calendar className="w-3.5 h-3.5 flex-shrink-0" />
            <span>
              {new Date(node.details.date).toLocaleDateString("fr-FR", {
                day: "numeric",
                month: "long",
                year: "numeric",
              })}
            </span>
          </div>
        )}

        {node.details?.emotion && (
          <div className="flex items-center gap-2 text-xs">
            <Heart className="w-3.5 h-3.5 text-rose-300 flex-shrink-0" />
            <span className="text-rose-200">{node.details.emotion}</span>
          </div>
        )}

        {node.details?.significance !== undefined && (
          <div className="flex items-center gap-2 text-xs">
            <Star className="w-3.5 h-3.5 text-amber-300 flex-shrink-0" />
            <span className="text-white/58">Importance émotionnelle</span>
            <span className="text-amber-200 font-semibold ml-auto">
              {(node.details.significance * 100).toFixed(0)}%
            </span>
          </div>
        )}

        {node.details?.tags && node.details.tags.length > 0 && (
          <div>
            <div className="flex items-center gap-1.5 mb-2 text-white/25">
              <Tag className="w-3 h-3" />
              <span className="text-[10px] uppercase tracking-wider">
                Étiquettes
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {node.details.tags.map((tag) => (
                <span
                  key={tag}
                  className="text-[10px] px-2 py-0.5 rounded-[var(--radius-full)] bg-white/8 text-white/62 border border-white/15"
                >
                  #{tag}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="mt-5 pt-4 border-t border-white/8">
        <div className="flex items-center gap-2">
          <div
            className="w-2.5 h-2.5 rounded-[var(--radius-full)]"
            style={{
              backgroundColor: node.color,
              boxShadow: `0 0 8px ${node.color}80`,
            }}
          />
          <span className="text-[10px] text-white/25 uppercase tracking-widest">
            {TYPE_LABELS[node.type]}
          </span>
        </div>
      </div>
    </motion.aside>
  );
}

// ─── Legend ───────────────────────────────────────────────────────────────────

const LEGEND = [
  { color: "#f7c874", label: "Patiente" },
  { color: "#8ea8ff", label: "Personnes" },
  { color: "#8ddfb5", label: "Aidants" },
  { color: "#8fdad2", label: "Lieux" },
  { color: "#95bbff", label: "Routines" },
  { color: "#f7a8c6", label: "Épisodes" },
];

// ─── Memory Graph ─────────────────────────────────────────────────────────────

export default function MemoryGraph() {
  const graphRef = useRef<ReturnType<typeof ForceGraph2D> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: 800, h: 600 });
  const [selected, setSelected] = useState<GraphNode | null>(null);

  // Deep-copy so force-graph can freely mutate positions
  const graphData = useMemo(
    () => ({
      nodes: MOCK_GRAPH.nodes.map((n) => ({ ...n })),
      links: MOCK_GRAPH.links.map((l) => ({ ...l })),
    }),
    []
  );

  // Responsive sizing
  useEffect(() => {
    const update = () => {
      if (containerRef.current) {
        setDims({
          w: containerRef.current.offsetWidth,
          h: containerRef.current.offsetHeight,
        });
      }
    };
    update();
    const ro = new ResizeObserver(update);
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Auto-fit after layout stabilizes
  useEffect(() => {
    const t = setTimeout(() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (graphRef.current as any)?.zoomToFit(500, 70);
    }, 700);
    return () => clearTimeout(t);
  }, []);

  // Custom node painter
  const paintNode = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (node: any, ctx: CanvasRenderingContext2D, scale: number) => {
      const { x, y, color, size, name, type } = node as GraphNode & {
        x: number;
        y: number;
      };
      if (x === undefined || y === undefined) return;

      const isPatient = type === "patient";
      const isSelected = selected?.id === node.id;
      const r = size as number;

      // Outer glow
      const grad = ctx.createRadialGradient(x, y, 0, x, y, r * 2.8);
      grad.addColorStop(0, `${color}30`);
      grad.addColorStop(1, `${color}00`);
      ctx.beginPath();
      ctx.arc(x, y, r * 2.8, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Selection ring
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(x, y, r + 4.5, 0, Math.PI * 2);
        ctx.strokeStyle = `${color}bb`;
        ctx.lineWidth = 1.5 / scale;
        ctx.stroke();
      }

      // Node fill
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = isPatient ? color : `${color}bb`;
      ctx.fill();

      // Patient outline
      if (isPatient) {
        ctx.strokeStyle = `${color}ff`;
        ctx.lineWidth = 1.8 / scale;
        ctx.stroke();
      }

      // Label
      const fontSize = Math.max(isPatient ? 10 : 8.5, (isPatient ? 10 : 8.5) / scale);
      ctx.font = `${isPatient ? 600 : 500} ${fontSize}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = isPatient
        ? "rgba(255,255,255,0.9)"
        : "rgba(255,255,255,0.65)";
      ctx.fillText(name as string, x, y + r + 2.5 / scale);
    },
    [selected]
  );

  const handleNodeClick = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (node: any) => {
      setSelected((prev) =>
        prev?.id === (node as GraphNode).id ? null : (node as GraphNode)
      );
    },
    []
  );

  return (
    <div className="w-full h-full relative overflow-hidden" ref={containerRef}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute top-0 left-0 z-10 px-6 pt-8 pb-4 pointer-events-none"
      >
        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-subtle mb-2">
          Graphe de mémoire
        </p>
        <h1 className="text-section text-white leading-tight">
          Le Cerveau
        </h1>
        <p className="text-sm text-muted mt-1">
          Réseau cognitif de Mamie Rose
        </p>
      </motion.div>

      {/* Legend */}
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
        className="absolute bottom-6 left-6 z-10 glass rounded-[var(--radius-md)] p-3 space-y-1.5"
      >
        {LEGEND.map(({ color, label }) => (
          <div key={label} className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-[var(--radius-full)] flex-shrink-0"
              style={{ backgroundColor: color }}
            />
            <span className="text-[10px] text-white/65">{label}</span>
          </div>
        ))}
      </motion.div>

      {/* Hint */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="absolute bottom-6 right-6 z-10 text-[10px] text-white/35 pointer-events-none"
      >
        Cliquez sur un nœud pour l'explorer
      </motion.div>

      {/* Graph canvas */}
      <div className="force-graph-container w-full h-full">
        <ForceGraph2D
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          ref={graphRef as any}
          graphData={graphData}
          width={dims.w}
          height={dims.h}
          backgroundColor="transparent"
          nodeLabel=""
          nodeCanvasObject={paintNode}
          nodeCanvasObjectMode={() => "replace"}
          linkColor={() => "rgba(142, 168, 255, 0.2)"}
          linkWidth={1}
          linkDirectionalParticles={2}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleColor={() => "rgba(142, 168, 255, 0.42)"}
          linkDirectionalParticleSpeed={0.004}
          onNodeClick={handleNodeClick}
          d3AlphaDecay={0.018}
          d3VelocityDecay={0.28}
          cooldownTicks={200}
          enableNodeDrag
          enableZoomInteraction
          nodeRelSize={1}
          minZoom={0.4}
          maxZoom={5}
        />
      </div>

      {/* Slide-in detail panel */}
      <AnimatePresence>
        {selected && (
          <NodeDetailPanel
            node={selected}
            onClose={() => setSelected(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

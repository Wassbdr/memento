// ─── Alert Types ─────────────────────────────────────────────────────────────

export type AlertLevel = "clinique" | "routine" | "emotionnel";

export interface Alert {
  id: string;
  level: AlertLevel;
  title: string;
  description: string;
  timestamp: string;
  tags: string[];
  relatedPerson?: string;
}

// ─── Graph Types ─────────────────────────────────────────────────────────────

export type NodeType = "patient" | "person" | "place" | "routine" | "episode";

export interface NodeDetails {
  subtitle?: string;
  notes?: string;
  date?: string;
  emotion?: string;
  significance?: number;
  tags?: string[];
}

export interface GraphNode {
  id: string;
  name: string;
  type: NodeType;
  color: string;
  size: number;
  details?: NodeDetails;
  // Injected by force-graph at runtime
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

export interface GraphLink {
  source: string;
  target: string;
  label: string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

// ─── Mock Alerts ─────────────────────────────────────────────────────────────

export const MOCK_ALERTS: Alert[] = [
  {
    id: "alert-1",
    level: "clinique",
    title: "Désorientée dans la cuisine",
    description:
      "Rose a semblé chercher ses lunettes dans la cuisine pendant plusieurs minutes. A exprimé une légère frustration avant d'être réorientée par Memento.",
    timestamp: "2026-04-08T07:42:00",
    tags: ["orientation", "domicile", "matin"],
  },
  {
    id: "alert-2",
    level: "emotionnel",
    title: "Inquiète pour le déjeuner de dimanche",
    description:
      "S'interroge si Claire va bien venir dimanche. A répété la question trois fois en dix minutes. Mémento a rappelé l'ancrage : « Claire arrive toujours après la matinée. »",
    timestamp: "2026-04-08T08:15:00",
    tags: ["famille", "anxiété", "répétition"],
    relatedPerson: "Claire Martin",
  },
  {
    id: "alert-3",
    level: "routine",
    title: "Médicaments du matin pris avec Sophie",
    description:
      "Sophie a préparé le pilulier. Rose a pris ses médicaments après le petit-déjeuner sans difficulté. Humeur sereine.",
    timestamp: "2026-04-08T09:03:00",
    tags: ["médicaments", "routine", "sophie"],
    relatedPerson: "Sophie Benali",
  },
  {
    id: "alert-4",
    level: "emotionnel",
    title: "A cherché ses clés",
    description:
      "Rose a cherché ses clés pendant environ 5 minutes. Léger moment de panique avant de les retrouver sur la table basse du salon.",
    timestamp: "2026-04-07T14:22:00",
    tags: ["objets", "anxiété", "domicile"],
  },
  {
    id: "alert-5",
    level: "clinique",
    title: "Confusion temporelle — jour de la semaine",
    description:
      "A demandé si c'était encore dimanche. Réorientation effectuée par Memento avec l'ancrage spatial : le fauteuil bleu près de la bibliothèque.",
    timestamp: "2026-04-07T16:55:00",
    tags: ["temps", "confusion", "réorientation"],
  },
  {
    id: "alert-6",
    level: "routine",
    title: "Petit-déjeuner terminé — humeur calme",
    description:
      "A pris son thé dans la tasse rouge avec deux biscottes et de la confiture d'abricot. Aucun incident. Humeur posée.",
    timestamp: "2026-04-08T08:22:00",
    tags: ["petit-déjeuner", "routine", "cuisine"],
  },
  {
    id: "alert-7",
    level: "emotionnel",
    title: "Se souvient avec joie de Lucas",
    description:
      "A évoqué spontanément le passage de Lucas mercredi dernier — fière qu'il ait réparé la télévision. Souvenir positif et ancré.",
    timestamp: "2026-04-08T10:30:00",
    tags: ["famille", "joie", "souvenir"],
    relatedPerson: "Lucas Martin",
  },
];

// ─── Mock Graph ───────────────────────────────────────────────────────────────

const C = {
  patient: "#f59e0b",
  person: "#a78bfa",
  caregiver: "#86efac",
  place: "#2dd4bf",
  routine: "#60a5fa",
  episode: "#f472b6",
};

export const MOCK_GRAPH: GraphData = {
  nodes: [
    // Patient — central, prominent
    {
      id: "rose",
      name: "Mamie Rose",
      type: "patient",
      color: C.patient,
      size: 14,
      details: {
        subtitle: "Rose Martin · Patiente",
        notes:
          "Vit dans son appartement rue des Lilas à Lyon. Alzheimer modéré, suivi à domicile.",
        tags: ["patiente", "alzheimer", "domicile"],
      },
    },
    // People
    {
      id: "claire",
      name: "Claire",
      type: "person",
      color: C.person,
      size: 9,
      details: {
        subtitle: "Claire Martin · Fille",
        notes:
          "Organise les rendez-vous médicaux, fait les courses et vient souvent le dimanche pour déjeuner.",
        significance: 0.98,
        tags: ["famille", "fille", "aidante-principale"],
      },
    },
    {
      id: "lucas",
      name: "Lucas",
      type: "person",
      color: C.person,
      size: 7,
      details: {
        subtitle: "Lucas Martin · Petit-fils",
        notes:
          "Passe le mercredi après les cours. Aide à régler la télévision. Rose est très fière de lui.",
        significance: 0.82,
        tags: ["famille", "petit-fils", "mercredi"],
      },
    },
    {
      id: "sophie",
      name: "Sophie",
      type: "person",
      color: C.caregiver,
      size: 7,
      details: {
        subtitle: "Sophie Benali · Aide à domicile",
        notes:
          "Vient le matin en semaine pour aider au petit-déjeuner et à la prise des médicaments.",
        significance: 0.76,
        tags: ["professionnel", "aide-domicile", "matin"],
      },
    },
    // Places
    {
      id: "salon",
      name: "Salon",
      type: "place",
      color: C.place,
      size: 6,
      details: {
        subtitle: "Salon · Domicile",
        notes:
          "Fauteuil bleu, bibliothèque en bois clair et photos de famille sur la commode. Lieu de repos privilégié.",
        tags: ["domicile", "repos", "ancrage"],
      },
    },
    {
      id: "cuisine",
      name: "Cuisine",
      type: "place",
      color: C.place,
      size: 6,
      details: {
        subtitle: "Cuisine · Domicile",
        notes:
          "Petite table ronde près de la fenêtre. La tasse rouge préférée de Rose est toujours à la même place.",
        tags: ["domicile", "repas", "routine"],
      },
    },
    {
      id: "bakery",
      name: "Boulangerie",
      type: "place",
      color: C.place,
      size: 5,
      details: {
        subtitle: "Boulangerie du coin · Quartier",
        notes:
          "À deux rues de l'appartement. Connue pour les tartelettes au citron. Sortie apaisante pour Rose.",
        tags: ["quartier", "sorties", "apaisement"],
      },
    },
    // Routines
    {
      id: "r-breakfast",
      name: "Petit-déjeuner",
      type: "routine",
      color: C.routine,
      size: 5,
      details: {
        subtitle: "Routine · Tous les jours à 08:00",
        notes:
          "Thé léger dans la tasse rouge avec deux biscottes et confiture d'abricot. La bouilloire comme signal.",
        tags: ["matin", "alimentation", "quotidien"],
      },
    },
    {
      id: "r-medication",
      name: "Médicaments",
      type: "routine",
      color: C.routine,
      size: 5,
      details: {
        subtitle: "Routine · Lundi–Vendredi à 09:00",
        notes:
          "Sophie prépare le pilulier après le petit-déjeuner. Le pilulier bleu posé à côté de la corbeille à pain.",
        tags: ["santé", "matin", "sophie"],
      },
    },
    {
      id: "r-sunday",
      name: "Déjeuner dimanche",
      type: "routine",
      color: C.routine,
      size: 5,
      details: {
        subtitle: "Routine · Le dimanche à 12:30",
        notes:
          "Claire vient partager le repas et apporte souvent un dessert. La nappe claire mise avant midi.",
        tags: ["famille", "dimanche", "claire"],
      },
    },
    // Episodes
    {
      id: "ep-tarte",
      name: "Tarte aux pommes",
      type: "episode",
      color: C.episode,
      size: 5,
      details: {
        subtitle: "Épisode · 29 mars 2026",
        notes:
          "Claire a apporté une tarte aux pommes et déjeuné avec Rose dans la cuisine. Repas calme et joyeux.",
        date: "2026-03-29",
        emotion: "Joie (0.92)",
        tags: ["famille", "repas", "dimanche"],
      },
    },
    {
      id: "ep-lucas",
      name: "Lucas & la télé",
      type: "episode",
      color: C.episode,
      size: 5,
      details: {
        subtitle: "Épisode · 1er avril 2026",
        notes:
          "Lucas est passé en fin d'après-midi, a réparé la télévision et bu un jus d'orange avec Rose dans le salon.",
        date: "2026-04-01",
        emotion: "Fierté (0.74)",
        tags: ["famille", "télévision", "mercredi"],
      },
    },
    {
      id: "ep-bakery",
      name: "Sortie boulangerie",
      type: "episode",
      color: C.episode,
      size: 5,
      details: {
        subtitle: "Épisode · 3 avril 2026",
        notes:
          "Sophie a accompagné Rose à la boulangerie pour acheter du pain frais et une tartelette au citron.",
        date: "2026-04-03",
        emotion: "Apaisement (0.68)",
        tags: ["sortie", "quartier", "sophie"],
      },
    },
  ],
  links: [
    // Rose ↔ People
    { source: "rose", target: "claire", label: "fille de" },
    { source: "rose", target: "lucas", label: "petit-fils de" },
    { source: "rose", target: "sophie", label: "aidante de" },
    // Rose ↔ Places
    { source: "rose", target: "salon", label: "vit dans" },
    { source: "rose", target: "cuisine", label: "vit dans" },
    // Rose ↔ Routines
    { source: "rose", target: "r-breakfast", label: "routine" },
    { source: "rose", target: "r-medication", label: "routine" },
    { source: "rose", target: "r-sunday", label: "routine" },
    // People ↔ Routines
    { source: "sophie", target: "r-medication", label: "prépare" },
    { source: "claire", target: "r-sunday", label: "participe" },
    // People ↔ Places
    { source: "claire", target: "cuisine", label: "visite" },
    { source: "lucas", target: "salon", label: "visite" },
    { source: "sophie", target: "bakery", label: "accompagne" },
    // Episodes
    { source: "claire", target: "ep-tarte", label: "présente" },
    { source: "ep-tarte", target: "cuisine", label: "lieu" },
    { source: "lucas", target: "ep-lucas", label: "présent" },
    { source: "ep-lucas", target: "salon", label: "lieu" },
    { source: "sophie", target: "ep-bakery", label: "présente" },
    { source: "ep-bakery", target: "bakery", label: "lieu" },
  ],
};

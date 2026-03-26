/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      animation: {
        "fade-in":      "fadeIn 0.4s ease-out both",
        "slide-up":     "slideUp 0.35s ease-out both",
        "slide-down":   "slideDown 0.4s ease-out both",
        "glow-pulse":   "glowPulse 2.5s ease-in-out infinite",
        "ring-ping-1":  "ringPing 2.2s ease-out infinite",
        "ring-ping-2":  "ringPing 2.2s ease-out 0.73s infinite",
        "ring-ping-3":  "ringPing 2.2s ease-out 1.46s infinite",
        "spin-slow":    "spin 14s linear infinite",
        "spin-reverse": "spinReverse 18s linear infinite",
        "float":        "float 7s ease-in-out infinite",
        "status-blink": "statusBlink 2s steps(2) infinite",
      },
      keyframes: {
        fadeIn: {
          from: { opacity: "0" },
          to:   { opacity: "1" },
        },
        slideUp: {
          from: { opacity: "0", transform: "translateY(10px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        slideDown: {
          from: { opacity: "0", transform: "translateY(-12px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        glowPulse: {
          "0%, 100%": { opacity: "0.65" },
          "50%":      { opacity: "1" },
        },
        ringPing: {
          "0%":   { transform: "scale(1)",    opacity: "0.65" },
          "100%": { transform: "scale(1.85)", opacity: "0" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%":      { transform: "translateY(-10px)" },
        },
        spinReverse: {
          from: { transform: "rotate(0deg)" },
          to:   { transform: "rotate(-360deg)" },
        },
        statusBlink: {
          "0%, 100%": { opacity: "1" },
          "50%":      { opacity: "0.3" },
        },
      },
    },
  },
  plugins: [],
};

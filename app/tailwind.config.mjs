/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // paleta dark — base do app
        ink: {
          50:  "#f5f7fa",
          100: "#e4e8ee",
          200: "#cbd2dd",
          300: "#a4afc1",
          400: "#7a8699",
          500: "#5b6577",
          600: "#454e5e",
          700: "#363d4a",
          800: "#262b35",
          900: "#1a1d24",
          950: "#0f1116",
        },
        accent: {
          DEFAULT: "#7c3aed",  // roxo Anthropic-ish
          fg:      "#a78bfa",
        },
      },
    },
  },
  plugins: [],
};

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Roboto', 'sans-serif'],    // Font mặc định cho toàn app
        roboto: ['Roboto', 'sans-serif'],  // Hoặc dùng className="font-roboto"
      },
    },
  },
  plugins: [],
}
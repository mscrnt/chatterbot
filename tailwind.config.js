/** Tailwind config — used only when you swap base.html away from the Play CDN
 * to a precompiled stylesheet. See README "Tailwind CLI build". */
module.exports = {
  content: [
    "./src/chatterbot/web/templates/**/*.html",
    "./src/chatterbot/web/static/js/**/*.js",
  ],
  theme: { extend: {} },
  plugins: [],
};

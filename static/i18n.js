/* Minimal i18n for the dashboard.

   English is the CANONICAL language: the markup and the JS render
   strings are English by construction. A locale file maps English
   text to the translation:
   static/_locales/ru/messages.json = { "Sequence": "Последовательность" }
   A future language = one more folder with the same flat map; en
   needs no file.

   Language resolution (first match wins):
   1. localStorage "lang" (Settings -> Interface language)
   2. navigator.language — ru* -> ru, anything else -> en

   Usage:
   - DOM:  i18n.apply() walks text nodes, title=, placeholder=,
           aria-label= and data-tip= and swaps those present in the map.
   - JS:   i18n.t("Run") -> "Запустить" (or the input when lang=en).

   The settings select re-seeds localStorage and reloads: dynamic
   strings are rendered from too many places for a live swap to be
   fully consistent. */

"use strict";

(function () {
  const LOCALES = ["en", "ru"];

  function detect() {
    const saved = localStorage.getItem("lang");
    if (saved && LOCALES.includes(saved)) return saved;
    const langs = navigator.languages || [navigator.language || "en"];
    for (const l of langs) {
      if (/^ru\b/i.test(l)) return "ru";
    }
    return "en";
  }

  const api = {
    lang: "en",
    locales: LOCALES,      // the settings page renders its select from this
    ready: null,
    t(text, ...args) {
      let out = (api.lang !== "en" && map[text]) || text;
      if (args.length) {
        out = out.replace(/\{(\d+)\}/g, (m, i) =>
          args[Number(i)] !== undefined ? String(args[Number(i)]) : m);
      }
      return out;
    },
    set(langCode) {
      if (!LOCALES.includes(langCode)) return;
      localStorage.setItem("lang", langCode);
      location.reload();
    },
    apply,
  };

  let map = {};     // en text -> translated text ({} for en)

  async function load() {
    const lang = detect();
    document.documentElement.lang = lang;
    if (lang !== "en") {
      try {
        const res = await fetch(`/static/_locales/${lang}/messages.json`);
        map = await res.json();
      } catch (e) {
        map = {};
        console.error("i18n: failed to load locale", e);
      }
    }
    api.lang = lang;              // en is canonical: nothing to fetch
    if (lang !== "en") apply();
  }

  function apply(root) {
    if (api.lang === "en") return;
    root = root || document;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const plain = n.nodeValue.trim();
      const translated = map[plain];
      if (translated) {
        n.nodeValue = translated;
      }
    }
    for (const attr of ["title", "placeholder", "aria-label", "data-tip"]) {
      for (const el of root.querySelectorAll(`[${attr}]`)) {
        const translated = map[el.getAttribute(attr)];
        if (translated) el.setAttribute(attr, translated);
      }
    }
  }

  window.i18n = api;
  api.ready = load();     // assigned AFTER the api object exists
})();

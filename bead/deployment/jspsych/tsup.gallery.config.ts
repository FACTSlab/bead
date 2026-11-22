import { defineConfig } from "tsup";

export default defineConfig([{
  entry: {
    "gallery-bundle": "src/gallery/gallery-bundle.ts",
  },
  format: ["iife"],
  globalName: "BeadGallery",
  dts: false,
  sourcemap: false,
  clean: false,
  target: "es2020",
  splitting: false,
  treeshake: true,
  minify: false,
  // Bead plugins depend only on local jsPsych type declarations.
  external: ["jspsych"],
  outDir: "dist",
  esbuildOptions(options) {
    options.banner = {
      js: "/* @bead/jspsych-gallery - Interactive demo bundle */",
    };
  },
}, {
  entry: { "slopit-bundle": "src/slopit/index.ts" },
  format: ["iife"],
  globalName: "SlopitJsPsych",
  dts: false,
  target: "es2020",
  outDir: "dist",
  noExternal: [/^@slopit\//, /^jspsych$/],
  minify: true,
}, {
  entry: { "list-distributor": "src/lib/list-distributor.ts" },
  format: ["iife"],
  globalName: "BeadLists",
  dts: false,
  target: "es2020",
  outDir: "dist",
}]);

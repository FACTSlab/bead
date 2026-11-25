// Copy the scoped package manager dependencies into the deployable browser runtime.
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const entryPoints = {
  jspsych: "jspsych.js",
  "@jspsych/plugin-browser-check": "browser-check.js",
  "@jspsych/plugin-fullscreen": "fullscreen.js",
  "@jspsych/plugin-html-button-response": "html-button-response.js",
  "@jspsych/plugin-html-keyboard-response": "html-keyboard-response.js",
  "@jspsych/plugin-instructions": "instructions.js",
  "@jspsych/plugin-survey-html-form": "survey-html-form.js",
  "@jspsych/plugin-survey": "survey.js",
};
const output = new URL("./dist/runtime/", import.meta.url);
await mkdir(output, { recursive: true });
const manifest = {};
async function copyAsset(name, source, packageName, version) {
  // Published source maps are unnecessary at runtime and are not deployed.
  let sourceText = (await readFile(source, "utf8")).replace(/(?:^\/\/#[ ]*sourceMappingURL=.*$|\/\*#[ ]*sourceMappingURL=.*?\*\/)/gm, "");
  if (name === "survey.css") {
    // Use the browser's locally installed fonts.
    sourceText = sourceText.replace(/@font-face\s*\{[^}]*https?:\/\/[^}]*\}/g, "");
  }
  await writeFile(new URL(name, output), sourceText, "utf8");
  manifest[name] = {
    package: packageName,
    version,
    sha256: createHash("sha256").update(sourceText).digest("hex"),
  };
}

for (const [packageName, asset] of Object.entries(entryPoints)) {
  const dist = dirname(require.resolve(packageName));
  const root = dirname(dist);
  const { version } = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
  await copyAsset(asset, join(dist, "index.browser.min.js"), packageName, version);
  if (packageName === "jspsych") {
    await copyAsset("jspsych.css", join(root, "css", "jspsych.css"), packageName, version);
  } else if (packageName === "@jspsych/plugin-survey") {
    await copyAsset("survey.css", join(root, "css", "survey.css"), packageName, version);
  }
}
await copyFile(new URL("./licenses/jspsych.txt", import.meta.url), new URL("jspsych.LICENSE", output));
await writeFile(new URL("manifest.json", output), `${JSON.stringify(manifest, null, 2)}\n`);

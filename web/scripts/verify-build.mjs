import { access, readFile } from "node:fs/promises";

const required = [
  "index.html",
  "styles.css",
  "app.js",
  "data/core.json",
  "data/careers.json",
  "data/playoffs.json",
];

await Promise.all(required.map((path) => access(path)));

const data = JSON.parse(await readFile("data/core.json", "utf8"));
if (!data.meta?.seasons?.length || !data.glossary) {
  throw new Error("The generated site data is incomplete.");
}

console.log(
  `Static site ready: ${data.meta.seasonCount} seasons, ` +
    `${data.meta.playerCount} players.`,
);

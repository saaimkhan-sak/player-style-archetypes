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

const [coreSource, appSource, stylesSource] = await Promise.all([
  readFile("data/core.json", "utf8"),
  readFile("app.js", "utf8"),
  readFile("styles.css", "utf8"),
]);
const data = JSON.parse(coreSource);
if (!data.meta?.seasons?.length || !data.glossary) {
  throw new Error("The generated site data is incomplete.");
}
if (
  !/\.bar-fill\s*\{[\s\S]*?display:\s*block;[\s\S]*?\}/.test(
    stylesSource,
  ) ||
  appSource.includes("profile.share / maxShare")
) {
  throw new Error("Season profile bars are not using visible, literal shares.");
}

const seasonPayloads = await Promise.all(
  data.meta.seasons.map(async ({ key }) => ({
    key,
    payload: JSON.parse(
      await readFile(`data/seasons/${key}.json`, "utf8"),
    ),
  })),
);
const seasonReadSignatures = new Set();
const seasonEditorials = new Set();
for (const { key, payload } of seasonPayloads) {
  for (const group of ["forwards", "defense"]) {
    const read = payload[group]?.seasonRead;
    if (
      !read?.headline ||
      read.paragraphs?.length < 2 ||
      read.facts?.length < 3
    ) {
      throw new Error(`${key} ${group} is missing its season read.`);
    }
    seasonReadSignatures.add(
      `${read.headline}\n${read.paragraphs.join("\n")}`,
    );
    seasonEditorials.add(read.paragraphs[1]);
    if (
      read.comparison !== null ||
      read.metrics?.profileCount !== payload[group].profiles.length
    ) {
      throw new Error(
        `${key} ${group} contains an invalid cross-season interpretation.`,
      );
    }
  }
}
if (
  seasonReadSignatures.size !== data.meta.seasonCount * 2 ||
  seasonEditorials.size !== data.meta.seasonCount * 2
) {
  throw new Error("Season reads must be unique for every season and group.");
}

console.log(
  `Static site ready: ${data.meta.seasonCount} seasons, ` +
    `${data.meta.playerCount} players, ` +
    `${seasonReadSignatures.size} unique season reads.`,
);

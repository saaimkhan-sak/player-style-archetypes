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
if (
  !appSource.includes("https://assets.nhle.com/mugs/nhl/") ||
  !appSource.includes("https://assets.nhle.com/logos/nhl/svg/") ||
  !appSource.includes("<span>Games played</span>") ||
  !/\.player-headshot\s*\{[\s\S]*?object-fit:\s*cover;[\s\S]*?\}/.test(
    stylesSource,
  ) ||
  !stylesSource.includes(
    ".player-headshot-frame.is-loaded .player-headshot-initials",
  ) ||
  !/\.team-logo-list\s*\{[\s\S]*?position:\s*static;[\s\S]*?grid-area:\s*logos;[\s\S]*?\}/.test(
    stylesSource,
  ) ||
  !/\.team-logo-frame\s*\{[\s\S]*?width:\s*48px;[\s\S]*?height:\s*48px;[\s\S]*?\}/.test(
    stylesSource,
  )
) {
  throw new Error(
    "Player profile cards are missing their corrected headshot, detached team logo, or games-played treatment.",
  );
}
if (
  !appSource.includes("function profileMentionMarkup(value, profileNames)") ||
  !appSource.includes("escapeHTML(text.slice(offset, index))") ||
  !appSource.includes(">${escapeHTML(name)}</strong>") ||
  !appSource.includes("profileMentionMarkup(paragraph, profileNames)") ||
  !/\.season-read-profile\s*\{[\s\S]*?var\(--profile\)[\s\S]*?\}/.test(
    stylesSource,
  )
) {
  throw new Error(
    "Season reads are missing safe, color-coded archetype mentions.",
  );
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
const seasonParagraphs = new Set();
for (const { key, payload } of seasonPayloads) {
  for (const group of ["forwards", "defense"]) {
    const read = payload[group]?.seasonRead;
    const groupProfileNames = payload[group].profiles.map(
      (profile) => profile.name,
    );
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
    if (
      read.paragraphs.some(
        (paragraph) =>
          !groupProfileNames.some((name) => paragraph.includes(name)),
      )
    ) {
      throw new Error(
        `${key} ${group} has a paragraph without a current archetype mention.`,
      );
    }
    if (
      read.paragraphs.some((paragraph) =>
        /model learned|account for|combine for|top[- ]three|\d+\.\d+%/i.test(
          paragraph,
        ),
      )
    ) {
      throw new Error(
        `${key} ${group} fell back to data-regurgitation copy.`,
      );
    }
    read.paragraphs.forEach((paragraph) => seasonParagraphs.add(paragraph));
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
  seasonParagraphs.size !== data.meta.seasonCount * 4
) {
  throw new Error(
    "Every season read must contain two unique editorial paragraphs.",
  );
}

console.log(
  `Static site ready: ${data.meta.seasonCount} seasons, ` +
    `${data.meta.playerCount} players, ` +
    `${seasonReadSignatures.size} unique season reads.`,
);

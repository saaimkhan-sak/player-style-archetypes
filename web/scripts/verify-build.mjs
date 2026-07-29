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
  !/\.player-identity\s*\{[\s\S]*?grid-template-areas:\s*"portrait copy logos";[\s\S]*?\}/.test(
    stylesSource,
  ) ||
  !/\.team-logo-list\s*\{[\s\S]*?align-self:\s*start;[\s\S]*?justify-self:\s*end;[\s\S]*?\}/.test(
    stylesSource,
  ) ||
  !/\.team-logo-frame\s*\{[\s\S]*?width:\s*48px;[\s\S]*?height:\s*48px;[\s\S]*?\}/.test(
    stylesSource,
  )
) {
  throw new Error(
    "Player profile cards are missing their corrected headshot, top-right team logo, or games-played treatment.",
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
if (
  !appSource.includes("<h2>Team Roster Construction</h2>") ||
  !appSource.includes(
    "A depth-chart view of the selected team using the 12 forwards or 8 defensemen with the most regular-season ice time.",
  ) ||
  !appSource.includes(
    "Whether the team identity is concentrated in stars or spread through depth.",
  ) ||
  !appSource.includes("function rosterPlayerHeadshot(player, season)") ||
  !appSource.includes("function rosterTeamLogo(team, season)") ||
  !appSource.includes("hydratePlayerAssets(teamViewElement)") ||
  !stylesSource.includes(".roster-unit-grid") ||
  !stylesSource.includes(".roster-player-photo-image") ||
  !stylesSource.includes(".roster-team-logo-image")
) {
  throw new Error(
    "Team roster construction is missing its Streamlit content or visual assets.",
  );
}
if (
  !appSource.includes(
    "<h2>Need Finder (find players who match a target archetype)</h2>",
  ) ||
  !appSource.includes(
    "A ranked list of players who best match a selected style profile.",
  ) ||
  !appSource.includes(
    "“Target similarity (%)” is the model’s estimated probability that the player belongs to that archetype.",
  ) ||
  !appSource.includes("<label for=\"need-team\">Exclude team (optional)</label>") ||
  !appSource.includes("<option value=\"\">(none)</option>") ||
  !appSource.includes("<label for=\"need-profile\">Target archetype</label>") ||
  !appSource.includes("<label for=\"need-games\">Min REG games</label>") ||
  !appSource.includes("const NEED_GAME_VALUES = [") ||
  !appSource.includes("80, 82,") ||
  !appSource.includes("aria-valuemax=\"82\"") ||
  !appSource.includes("const minGames = needGamesValue(games)") ||
  !appSource.includes("const matches = eligible.slice(0, 80)") ||
  !appSource.includes("Number(b.points || 0) - Number(a.points || 0)") ||
  !appSource.includes("hydratePlayerAssets(results)") ||
  !stylesSource.includes(".need-finder-intro") ||
  !stylesSource.includes(".need-player-detail") ||
  !stylesSource.includes(".need-similarity-track")
) {
  throw new Error(
    "Need Finder is missing its Streamlit content, exact ranking logic, or responsive scouting presentation.",
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
    const needFinder = payload[group]?.needFinder;
    const needTargets = needFinder?.targets || [];
    const targetProfiles = needTargets.map((target) => target.profile);
    const maxTargetCluster = Math.max(
      ...needTargets.map((target) => Number(target.cluster)),
    );
    if (
      !needTargets.length ||
      new Set(targetProfiles).size !== targetProfiles.length ||
      !needFinder?.details ||
      needTargets.some(
        (target) =>
          !target.profile ||
          !Number.isInteger(Number(target.cluster)) ||
          !needFinder.details[String(target.cluster)],
      )
    ) {
      throw new Error(
        `${key} ${group} is missing Need Finder target metadata.`,
      );
    }
    for (const player of payload[group].players) {
      const requiredPlayerFields = [
        "regAtoi",
        "playoffGames",
        "playoffAtoi",
        "playoffPoints",
        "playoffGoals",
        "playoffAssists",
        "playoffShots",
        "playoffPlusMinus",
        "playoffPim",
        "needOrder",
      ];
      if (
        !Array.isArray(player.targetScores) ||
        player.targetScores.length <= maxTargetCluster ||
        player.targetScores.some(
          (score) =>
            !Number.isFinite(Number(score)) ||
            Number(score) < 0 ||
            Number(score) > 100,
        ) ||
        requiredPlayerFields.some((field) => !(field in player)) ||
        !/^\d{2,}:\d{2}$/.test(player.regAtoi) ||
        !/^\d{2,}:\d{2}$/.test(player.playoffAtoi)
      ) {
        throw new Error(
          `${key} ${group} has incomplete Need Finder player data.`,
        );
      }
    }
    const constructions = payload[group]?.teamConstructions;
    const expectedRosterSize = group === "forwards" ? 12 : 8;
    const unitSize = group === "forwards" ? 3 : 2;
    if (!constructions || Object.keys(constructions).length < 20) {
      throw new Error(
        `${key} ${group} is missing team roster construction data.`,
      );
    }
    for (const [team, construction] of Object.entries(constructions)) {
      const units = construction.units || [];
      const rosterSize = units.reduce(
        (total, unit) => total + (unit.players?.length || 0),
        0,
      );
      if (
        units.length < 3 ||
        units.length > 4 ||
        rosterSize < unitSize * 3 ||
        rosterSize > expectedRosterSize ||
        units.some(
          (unit) =>
            !unit.players?.length ||
            unit.players.length > unitSize ||
            unit.players.some(
              (player) =>
                !player.id ||
                !player.name ||
                !player.profile,
            ),
        ) ||
        !construction.dominant?.profile ||
        !construction.mix?.length ||
        !construction.gaps?.length
      ) {
        throw new Error(
          `${key} ${group} ${team} has an incomplete roster construction.`,
        );
      }
      if (
        construction.usesMoneyPuck &&
        !units.some(
          (unit) =>
            unit.fromCombination &&
            Number.isFinite(Number(unit.xgPct)),
        )
      ) {
        throw new Error(
          `${key} ${group} ${team} is missing MoneyPuck unit metadata.`,
        );
      }
    }
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

const latest = seasonPayloads.find(({ key }) => key === "20252026");
const ana = latest?.payload?.forwards?.teamConstructions?.ANA;
if (
  !ana ||
  ana.dominant.profile !== "High-Touch Risk/Reward Scorer" ||
  ana.dominant.overall !== 61.1 ||
  ana.dominant.top !== 67.5 ||
  ana.dominant.bottom !== 50.6 ||
  ana.units?.[0]?.players?.map((player) => player.name).join("|") !==
    "Chris Kreider|Leo Carlsson|Troy Terry" ||
  ana.units?.[0]?.minutes !== 469.5 ||
  ana.units?.[0]?.xgPct !== 0.591
) {
  throw new Error(
    "The Streamlit-parity ANA roster construction fixture changed unexpectedly.",
  );
}
const latestForwards = latest?.payload?.forwards;
const latestNeedTarget = latestForwards?.needFinder?.targets?.[0];
const latestNeedMatches = latestForwards?.players
  ?.filter((player) => Number(player.games) >= 20)
  .map((player) => ({
    ...player,
    similarity: Number(
      player.targetScores?.[Number(latestNeedTarget?.cluster)] || 0,
    ),
  }))
  .sort(
    (left, right) =>
      right.similarity - left.similarity ||
      Number(right.points || 0) - Number(left.points || 0) ||
      Number(left.needOrder || 0) - Number(right.needOrder || 0),
  )
  .slice(0, 80);
if (
  latestNeedTarget?.profile !== "Two-Way Shot-Share Driver" ||
  latestNeedTarget?.cluster !== 0 ||
  latestNeedMatches?.length !== 80 ||
  latestNeedMatches
    .slice(0, 5)
    .map((player) => player.name)
    .join("|") !==
    "Wyatt Johnston|Steven Stamkos|Adam Fantilli|Will Smith|Matty Beniers" ||
  latestNeedMatches.slice(0, 5).some(
    (player) => player.similarity !== 100,
  )
) {
  throw new Error(
    "The Streamlit-parity Need Finder fixture changed unexpectedly.",
  );
}

console.log(
  `Static site ready: ${data.meta.seasonCount} seasons, ` +
    `${data.meta.playerCount} players, ` +
    `${seasonReadSignatures.size} unique season reads.`,
);

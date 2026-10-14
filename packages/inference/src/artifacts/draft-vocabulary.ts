// A frequency-ranked draft vocabulary: `draft_vocab.json`, `{"ids": [...]}`, the
// target token ids a drafter may propose (FR-Spec, Zhao et al. 2025). The Qwen3.8
// Trellis artifact ships one in its MTP companion directory (`mtp/`); the
// companion's drafts and, where a graph's composition declares it, a tapped
// drafter's select over those rows of the target head.

import { readFileSync } from "node:fs";

export const DRAFT_VOCAB_FILE = "draft_vocab.json";

const lists = new Map<string, readonly number[]>();

/** The list at `path`: distinct ids in ascending order, read once per path.
 *  Throws when the file holds no ids, an id is not a nonnegative integer, or one
 *  falls outside the target's `vocabSize`. */
export function readDraftVocabulary(path: string, vocabSize: number): readonly number[] {
  let ids = lists.get(path);
  if (!ids) {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as { ids?: unknown };
    if (!Array.isArray(parsed.ids) || parsed.ids.length === 0)
      throw new Error(`draft vocabulary ${path}: expected {"ids":[...]}`);
    const sorted = [...new Set(parsed.ids as number[])].sort((a, b) => a - b);
    if (!sorted.every(id => Number.isInteger(id) && id >= 0))
      throw new Error(`draft vocabulary ${path}: ids must be nonnegative integers`);
    lists.set(path, ids = Object.freeze(sorted));
  }
  if (ids.at(-1)! >= vocabSize)
    throw new Error(`draft vocabulary ${path}: id ${ids.at(-1)} outside the target vocabulary ${vocabSize}`);
  return ids;
}

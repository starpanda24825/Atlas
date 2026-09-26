import type { ReactNode } from "react";

/**
 * A small markdown renderer.
 *
 * Deliberately dependency-free and never uses dangerouslySetInnerHTML: vault
 * notes and research reports are rendered from text that Atlas wrote, but
 * treating them as trusted HTML would still be a mistake.
 */

function inline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\n]+\*|\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let index = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) nodes.push(text.slice(last, match.index));
    const token = match[0];
    const key = `${keyPrefix}-i${index++}`;
    if (token.startsWith("`")) {
      nodes.push(
        <code key={key} className="rounded bg-slate-800 px-1 py-0.5 text-[0.85em] text-sky-300">
          {token.slice(1, -1)}
        </code>,
      );
    } else if (token.startsWith("**")) {
      nodes.push(
        <strong key={key} className="font-semibold text-slate-100">
          {token.slice(2, -2)}
        </strong>,
      );
    } else if (token.startsWith("[")) {
      const link = /\[([^\]]+)\]\(([^)]+)\)/.exec(token);
      if (link) {
        nodes.push(
          <a
            key={key}
            href={link[2]}
            target="_blank"
            rel="noreferrer"
            className="text-sky-400 underline decoration-sky-800 hover:decoration-sky-400"
          >
            {link[1]}
          </a>,
        );
      }
    } else {
      nodes.push(
        <em key={key} className="italic text-slate-300">
          {token.slice(1, -1)}
        </em>,
      );
    }
    last = match.index + token.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

const HEADING_CLASS: Record<number, string> = {
  1: "mt-4 text-lg font-semibold text-slate-100",
  2: "mt-4 text-base font-semibold text-slate-100",
  3: "mt-3 text-sm font-semibold text-slate-200",
  4: "mt-3 text-sm font-medium text-slate-200",
  5: "mt-2 text-xs font-semibold uppercase tracking-wide text-slate-400",
  6: "mt-2 text-xs font-semibold uppercase tracking-wide text-slate-400",
};

export function Markdown({ text }: { text: string }) {
  const lines = (text ?? "").replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let index = 0;
  let key = 0;

  while (index < lines.length) {
    const line = lines[index];

    // Fenced code block.
    if (line.trimStart().startsWith("```")) {
      const body: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trimStart().startsWith("```")) {
        body.push(lines[index]);
        index += 1;
      }
      index += 1; // consume closing fence
      blocks.push(
        <pre
          key={`b${key++}`}
          className="my-3 overflow-x-auto rounded-md border border-slate-800 bg-slate-950 p-3 text-xs leading-relaxed text-slate-300"
        >
          <code>{body.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    // Headings.
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      blocks.push(
        <div key={`b${key++}`} className={HEADING_CLASS[level]}>
          {inline(heading[2], `h${key}`)}
        </div>,
      );
      index += 1;
      continue;
    }

    // Horizontal rule.
    if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) {
      blocks.push(<hr key={`b${key++}`} className="my-4 border-slate-800" />);
      index += 1;
      continue;
    }

    // Unordered list.
    if (/^\s*[-*+]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*[-*+]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*[-*+]\s+/, ""));
        index += 1;
      }
      blocks.push(
        <ul key={`b${key++}`} className="my-2 list-disc space-y-1 pl-5 text-sm text-slate-300">
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>{inline(item, `ul${key}-${itemIndex}`)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    // Ordered list.
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*\d+[.)]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*\d+[.)]\s+/, ""));
        index += 1;
      }
      blocks.push(
        <ol key={`b${key++}`} className="my-2 list-decimal space-y-1 pl-5 text-sm text-slate-300">
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>{inline(item, `ol${key}-${itemIndex}`)}</li>
          ))}
        </ol>,
      );
      continue;
    }

    // Blockquote.
    if (line.trimStart().startsWith(">")) {
      const quote: string[] = [];
      while (index < lines.length && lines[index].trimStart().startsWith(">")) {
        quote.push(lines[index].replace(/^\s*>\s?/, ""));
        index += 1;
      }
      blocks.push(
        <blockquote
          key={`b${key++}`}
          className="my-2 border-l-2 border-slate-600 pl-3 text-sm italic text-slate-400"
        >
          {inline(quote.join(" "), `q${key}`)}
        </blockquote>,
      );
      continue;
    }

    // Blank line.
    if (line.trim() === "") {
      index += 1;
      continue;
    }

    // Paragraph (until a blank line or a new block start).
    const paragraph: string[] = [];
    while (
      index < lines.length &&
      lines[index].trim() !== "" &&
      !/^(#{1,6})\s+/.test(lines[index]) &&
      !/^\s*[-*+]\s+/.test(lines[index]) &&
      !/^\s*\d+[.)]\s+/.test(lines[index]) &&
      !lines[index].trimStart().startsWith("```") &&
      !lines[index].trimStart().startsWith(">")
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push(
      <p key={`b${key++}`} className="my-2 text-sm leading-relaxed text-slate-300">
        {inline(paragraph.join(" "), `p${key}`)}
      </p>,
    );
  }

  return <div className="atlas-markdown">{blocks}</div>;
}

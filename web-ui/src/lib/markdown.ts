import { marked } from "marked";
import hljs from "highlight.js/lib/common";
import DOMPurify from "dompurify";

marked.setOptions({
  breaks: true,
  gfm: true,
});

export function renderMarkdown(src: string): string {
  const raw = marked.parse(src ?? "", { async: false }) as string;
  return DOMPurify.sanitize(raw, { ADD_ATTR: ["target"] });
}

// Highlight every <pre><code> inside a container, in small batches to keep the
// main thread responsive when a single message contains many code blocks.
export function highlightWithin(root: HTMLElement): void {
  const blocks = Array.from(root.querySelectorAll<HTMLElement>("pre code"));
  if (blocks.length === 0) return;
  if (blocks.length <= 3) {
    for (const b of blocks) hljs.highlightElement(b);
    return;
  }
  let idx = 0;
  const step = () => {
    for (let i = 0; i < 3 && idx < blocks.length; i++, idx++) hljs.highlightElement(blocks[idx]);
    if (idx < blocks.length) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

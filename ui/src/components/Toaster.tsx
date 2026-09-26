import * as RadixToast from "@radix-ui/react-toast";
import { useToastStore } from "../store/toasts";
import { cx } from "../lib/format";

export function Toaster() {
  const items = useToastStore((state) => state.items);
  const dismiss = useToastStore((state) => state.dismiss);

  return (
    <RadixToast.Provider swipeDirection="right" duration={6000}>
      {items.map((item) => (
        <RadixToast.Root
          key={item.id}
          defaultOpen
          onOpenChange={(open) => {
            if (!open) dismiss(item.id);
          }}
          className={cx(
            "pointer-events-auto rounded-lg border px-4 py-3 shadow-xl backdrop-blur",
            item.tone === "error"
              ? "border-rose-800 bg-rose-950/90"
              : "border-slate-700 bg-slate-900/95",
          )}
        >
          <RadixToast.Title className="text-sm font-medium text-slate-100">
            {item.title}
          </RadixToast.Title>
          {item.description ? (
            <RadixToast.Description className="mt-1 text-xs text-slate-400">
              {item.description}
            </RadixToast.Description>
          ) : null}
          <RadixToast.Close className="absolute right-2 top-2 text-slate-500 hover:text-slate-300">
            ×
          </RadixToast.Close>
        </RadixToast.Root>
      ))}
      <RadixToast.Viewport className="fixed bottom-4 right-4 z-[100] flex w-80 flex-col gap-2 outline-none" />
    </RadixToast.Provider>
  );
}

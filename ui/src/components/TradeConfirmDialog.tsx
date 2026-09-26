import * as Dialog from "@radix-ui/react-dialog";
import { useMutation } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";
import { api } from "../lib/api";
import { useAtlasStore } from "../store/atlas";
import { toast } from "../store/toasts";

function primitiveEntries(trade: Record<string, unknown>): Array<[string, string]> {
  return Object.entries(trade)
    .filter(([, value]) => value !== null && typeof value !== "object")
    .map(([key, value]) => [key.replace(/_/g, " "), String(value)]);
}

export function TradeConfirmDialog() {
  const trade = useAtlasStore((state) => state.tradeConfirmation);
  const setTradeConfirmation = useAtlasStore((state) => state.setTradeConfirmation);
  const pushActivity = useAtlasStore((state) => state.pushActivity);

  const tradeId = String(trade?.trade_id ?? trade?.id ?? trade?.order_id ?? "");

  const confirm = useMutation({
    mutationFn: (confirmed: boolean) => api.confirmTrade(tradeId, confirmed),
    onSuccess: (_result, confirmed) => {
      pushActivity("trade", `Trade ${tradeId} ${confirmed ? "confirmed" : "cancelled"}`);
      toast.info(confirmed ? "Trade submitted" : "Trade cancelled", tradeId);
      setTradeConfirmation(null);
    },
    onError: (error: Error) => toast.error("Trade action failed", error.message),
  });

  const open = trade !== null;

  return (
    <Dialog.Root open={open} onOpenChange={(next) => (!next && !confirm.isPending ? setTradeConfirmation(null) : undefined)}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[26rem] max-w-[90vw] -translate-x-1/2 -translate-y-1/2 rounded-xl border border-slate-700 bg-slate-900 p-5 shadow-2xl">
          <Dialog.Title className="flex items-center gap-2 text-base font-semibold text-slate-100">
            <AlertTriangle size={18} className="text-amber-400" />
            Confirm trade
          </Dialog.Title>
          <Dialog.Description className="mt-1 text-xs text-slate-400">
            Atlas is waiting on your approval before this order is placed.
          </Dialog.Description>

          <dl className="mt-4 max-h-64 space-y-1.5 overflow-auto rounded-lg border border-slate-800 bg-slate-950/60 p-3">
            {trade && primitiveEntries(trade).length > 0 ? (
              primitiveEntries(trade).map(([key, value]) => (
                <div key={key} className="flex justify-between gap-4 text-xs">
                  <dt className="capitalize text-slate-500">{key}</dt>
                  <dd className="text-right font-medium text-slate-200">{value}</dd>
                </div>
              ))
            ) : (
              <p className="text-xs text-slate-500">No trade details available.</p>
            )}
          </dl>

          <div className="mt-5 flex justify-end gap-2">
            <Dialog.Close asChild>
              <button
                type="button"
                disabled={confirm.isPending}
                onClick={() => confirm.mutate(false)}
                className="rounded-md border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-50"
              >
                Cancel
              </button>
            </Dialog.Close>
            <button
              type="button"
              disabled={confirm.isPending || !tradeId}
              onClick={() => confirm.mutate(true)}
              className="rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
            >
              Confirm Trade
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

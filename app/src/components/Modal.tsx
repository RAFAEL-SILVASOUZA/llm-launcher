import { X } from "lucide-react";
import { ReactNode, useEffect } from "react";

type Props = {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
  width?: string;
  footer?: ReactNode;
  closable?: boolean;
};

export function Modal({ open, onClose, title, children, width = "max-w-3xl", footer, closable = true }: Props) {
  useEffect(() => {
    if (!open || !closable) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", h);
    return () => document.removeEventListener("keydown", h);
  }, [open, onClose, closable]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      onClick={closable ? onClose : undefined}
    >
      <div
        className={`${width} w-full bg-ink-900 border border-ink-700 rounded-lg shadow-2xl flex flex-col max-h-[92vh]`}
        onClick={(e) => e.stopPropagation()}
      >
        {title && (
          <div className="px-5 py-3 border-b border-ink-800 flex items-center justify-between">
            <h2 className="font-medium text-ink-100">{title}</h2>
            {closable && (
              <button
                onClick={onClose}
                className="text-ink-400 hover:text-ink-100"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </div>
        )}
        <div className="flex-1 overflow-auto">{children}</div>
        {footer && (
          <div className="px-5 py-3 border-t border-ink-800 flex justify-end gap-2">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

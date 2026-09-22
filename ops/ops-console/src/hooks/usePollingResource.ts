import { useEffect, useRef, useState } from "react";

interface UsePollingResourceOptions<T> {
  load: () => Promise<T>;
  pollMs: number;
  enabled?: boolean;
  deps?: ReadonlyArray<unknown>;
  initialData?: T | null;
}

interface PollingResourceState<T> {
  data: T | null;
  loading: boolean;
  error: string;
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "unknown error";
}

export function usePollingResource<T>({
  load,
  pollMs,
  enabled = true,
  deps = [],
  initialData = null,
}: UsePollingResourceOptions<T>): PollingResourceState<T> {
  const [data, setData] = useState<T | null>(initialData);
  const [loading, setLoading] = useState<boolean>(enabled);
  const [error, setError] = useState("");
  const loadRef = useRef(load);

  useEffect(() => {
    loadRef.current = load;
  }, [load]);

  useEffect(() => {
    if (!enabled) {
      setData(initialData);
      setError("");
      setLoading(false);
      return;
    }

    let cancelled = false;

    const read = async (showLoading: boolean) => {
      if (showLoading) {
        setLoading(true);
      }
      try {
        const payload = await loadRef.current();
        if (cancelled) {
          return;
        }
        setData(payload);
        setError("");
      } catch (err) {
        if (cancelled) {
          return;
        }
        setError(toErrorMessage(err));
      } finally {
        if (!cancelled && showLoading) {
          setLoading(false);
        }
      }
    };

    void read(true);
    const timer = window.setInterval(() => {
      void read(false);
    }, pollMs);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [enabled, initialData, pollMs, ...deps]);

  return {
    data,
    loading,
    error,
  };
}

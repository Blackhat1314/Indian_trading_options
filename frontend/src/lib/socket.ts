import { wsUrl } from "./api";
import { useLive } from "./store";

/** Single websocket to the gateway with auto-reconnect and topic subscriptions.
 *  The gateway coalesces snapshot topics server-side, so the browser always gets the newest frame. */
class LiveSocket {
  private ws: WebSocket | null = null;
  private topics = new Set<string>();
  private retry = 0;
  private pingTimer: number | undefined;

  connect() {
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) return; // already connecting/open (StrictMode double-mount)
    useLive.getState().setConn("connecting");
    const ws = new WebSocket(wsUrl());
    this.ws = ws;
    ws.onopen = () => {
      this.retry = 0;
      useLive.getState().setConn("open");
      if (this.topics.size) this.send({ op: "sub", topics: [...this.topics] });
      window.clearInterval(this.pingTimer);
      this.pingTimer = window.setInterval(() => this.send({ op: "ping" }), 5000);
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data as string) as { topic: string; data: unknown; event?: boolean };
        useLive.getState().ingest(msg.topic, msg.data, msg.event);
      } catch {
        /* ignore malformed */
      }
    };
    ws.onclose = (ev) => {
      useLive.getState().setConn("closed");
      window.clearInterval(this.pingTimer);
      if (ev.code === 4401) {
        useLive.getState().setConn("denied"); // bad / missing access key — don't hammer the server
        return;
      }
      const delay = Math.min(10000, 500 * 2 ** this.retry++);
      window.setTimeout(() => this.connect(), delay);
    };
  }

  private send(obj: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj));
  }

  sub(topics: string[]) {
    const fresh = topics.filter((t) => !this.topics.has(t));
    fresh.forEach((t) => this.topics.add(t));
    if (fresh.length) this.send({ op: "sub", topics: fresh });
  }

  unsub(topics: string[]) {
    const gone = topics.filter((t) => this.topics.has(t));
    gone.forEach((t) => this.topics.delete(t));
    if (gone.length) this.send({ op: "unsub", topics: gone });
  }

  depth(sid: number) {
    this.send({ op: "depth", sid });
  }
}

export const live = new LiveSocket();

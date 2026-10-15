import { Component, type ErrorInfo, type ReactNode } from "react";
import { Panel } from "./ui";

/** Keeps a desk's render error inside that desk: without it React unmounts the whole terminal (blank page).
 *  Mount it with key={desk} so switching desks starts clean. */
export class DeskBoundary extends Component<{ desk: string; children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`desk "${this.props.desk}" failed to render`, error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="desk" style={{ display: "grid", padding: 6 }}>
        <Panel code="ERR" title={`${this.props.desk.toUpperCase()} desk hit an error`}>
          <div role="alert" style={{ padding: "14px 16px", display: "grid", gap: 10, maxWidth: 760 }}>
            <p style={{ margin: 0 }}>
              This desk could not draw the data it received. The other desks keep working — switch with the tabs or
              F-keys, or try this one again.
            </p>
            <code className="small muted" style={{ whiteSpace: "pre-wrap" }}>{error.message}</code>
            <div>
              <button type="button" className="btn primary" onClick={() => this.setState({ error: null })}>Try again</button>
            </div>
          </div>
        </Panel>
      </div>
    );
  }
}

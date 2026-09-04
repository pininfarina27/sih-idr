import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught application error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 my-8 max-w-xl mx-auto bg-white border border-red-200 rounded-xl shadow-md text-center">
          <div className="w-12 h-12 rounded-full bg-red-100 text-red-600 flex items-center justify-center mx-auto mb-4 text-xl font-bold">
            !
          </div>
          <h3 className="text-lg font-bold text-gray-800 mb-2">Something went wrong rendering this view</h3>
          <p className="text-sm text-gray-500 mb-4">
            {this.state.error?.message || "An unexpected error occurred while loading navigation tracks."}
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-semibold shadow transition-all"
          >
            Reload Application
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

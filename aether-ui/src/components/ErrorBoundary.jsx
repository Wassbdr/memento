import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    console.error("ui_crash", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            height: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            gap: "8px",
            background: "#040810",
            color: "#e4e4e7",
            fontFamily: "Inter, sans-serif",
          }}
        >
          <strong>Erreur d’affichage</strong>
          <span style={{ color: "#a1a1aa", fontSize: "13px" }}>
            Rechargez la page pour continuer.
          </span>
        </div>
      );
    }

    return this.props.children;
  }
}

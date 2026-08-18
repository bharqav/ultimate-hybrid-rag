from config.logging import setup_logging
from core.deps import App, ComposeResult, Footer, Header, Log, Static

log = setup_logging()


if App:

    class RAGDashboard(App):
        CSS = """
        #latency { height: 5; }
        #log { height: 80%; }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Latency: N/A", id="latency")
            yield Log(id="log")
            yield Footer()

        def on_mount(self):
            self.log("Ultimate RAG Dashboard ready")
            self.set_interval(5, self.update_stats)

        def update_stats(self):
            self.query_one("#latency").update("Metrics: (press Ctrl+C to quit)")

        def show_results(self, query, results):
            self.query_one("#log").write(f"Query: {query}")
            for i, r in enumerate(results):
                self.query_one("#log").write(
                    f"  {i + 1}. [{r.chunk.source_document}:{r.chunk.page_number}] fused={r.fused_score:.3f}"
                )

else:
    RAGDashboard = None  # type: ignore

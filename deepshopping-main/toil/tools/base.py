from langchain_core.tools import BaseTool


class AsyncTool(BaseTool):
    def _run(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError
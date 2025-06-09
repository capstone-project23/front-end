import chardet
import httpx
from bs4 import BeautifulSoup
from httpx._decoders import TextDecoder
from langchain.schema import Document
from markdownify import markdownify
from readabilipy import simple_json_from_html_string
from readabilipy.simple_json import have_node


def _clean_url(url: str) -> str:
    return url.strip("\"'")


def _parse_content_type_header(header: str) -> tuple[str, dict]:
    tokens = header.split(";")
    content_type, params = tokens[0].strip(), tokens[1:]
    params_dict = {}
    items_to_strip = "\"' "

    for param in params:
        param = param.strip()
        if param:
            key, value = param, True
            index_of_equals = param.find("=")
            if index_of_equals != -1:
                key = param[:index_of_equals].strip(items_to_strip)
                value = param[index_of_equals + 1 :].strip(items_to_strip)
            params_dict[key.lower()] = value
    return content_type, params_dict


class ContentTypeError(Exception):
    pass


class NoSyncChainMixin:
    def _call(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError


class HTTPXMixin:
    async def aget(
        self, url: str, headers: dict = dict(), content_type: str = None, **kwargs
    ) -> str:
        async with httpx.AsyncClient(
            timeout=10,
            follow_redirects=True,
            headers=headers,
        ) as client:
            async with client.stream("GET", _clean_url(url), **kwargs) as stream:
                stream.raise_for_status()
                if (
                    content_type is not None
                    and content_type not in stream.headers["Content-Type"]
                ):
                    raise ContentTypeError

                content = await anext(stream.aiter_bytes(2 * 1024 * 1024))
                content_type, params = _parse_content_type_header(
                    stream.headers["content-type"]
                )

                if "charset" in params:
                    encoding = params["charset"].strip("'\"")
                else:
                    encoding = chardet.detect(content).get("encoding") or "utf-8"

                decoder = TextDecoder(encoding=encoding)
                return "".join([decoder.decode(content), decoder.flush()])


class HTMLToMarkdownMixin:
    async def aclean_html(
        self,
        html_text: Document,
    ) -> Document:
        metadata = html_text.metadata
        md_text = self._transform_html_readability_markdown(html_text.page_content)
        if not md_text:
            metadata.update({"error": "No content found in HTMLToMarkdownMixin."})
            return Document(
                page_content="콘텐츠가 존재하지 않습니다.",
                metadata=metadata,
            )
        return Document(page_content=md_text, metadata=metadata)

    def _transform_html_readability_markdown(self, html_text: str) -> str:
        try:
            if have_node():
                readability_content = simple_json_from_html_string(
                    html_text,
                    use_readability=True,
                )
                plain_content = readability_content["plain_content"]
            else:
                doc = Document(html_text)
                plain_content = doc.summary()

            if plain_content is None:
                raise ValueError("No content extracted")

        except Exception:
            # BeautifulSoup fallback
            soup = BeautifulSoup(html_text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            plain_content = soup.get_text()

        md_content = markdownify(plain_content, strip=["a", "img"], heading_style="ATX")
        return md_content

    async def aget_title(
        self, url: str, headers=dict(), content_type=None, **kwargs
    ) -> str:
        async with httpx.AsyncClient(
            timeout=10,
            follow_redirects=True,
            headers=headers,
        ) as client:
            async with client.stream("GET", _clean_url(url), **kwargs) as stream:
                stream.raise_for_status()
                if (
                    content_type is not None
                    and content_type not in stream.headers["Content-Type"]
                ):
                    raise ContentTypeError

                content = await anext(stream.aiter_bytes(1 * 1024 * 1024))

                soup = BeautifulSoup(content.decode("utf-8"), "html.parser")
                # title_tag = await remove_title_tag(soup.find("title"))
                title_tag = soup.find("title").get_text(strip=True)

        return title_tag or None

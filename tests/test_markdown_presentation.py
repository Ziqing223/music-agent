"""P18-S1 markdown presentation tests: correct rendering + the security wall.

The security contract: no input may ever produce markup outside the
whitelist (p/br/strong/em/ul/ol/li/hr/code/a) and no input may execute as a
script -- every character that is not whitelisted-generated markup appears
only as visible escaped text.
"""

from __future__ import annotations

import unittest

from music_agent.markdown_presentation import render_assistant_markdown


class AssistantMarkdownTest(unittest.TestCase):
    """Rendering correctness for the supported subset."""

    def test_empty_text_renders_nothing(self) -> None:
        self.assertEqual(render_assistant_markdown(""), "")
        self.assertEqual(render_assistant_markdown("   \n\n  "), "")

    def test_single_paragraph(self) -> None:
        self.assertEqual(render_assistant_markdown("你好，世界"), "<p>你好，世界</p>")

    def test_multiple_paragraphs(self) -> None:
        html = render_assistant_markdown("第一段\n\n第二段")
        self.assertEqual(html, "<p>第一段</p><p>第二段</p>")

    def test_line_breaks_inside_paragraph(self) -> None:
        self.assertEqual(render_assistant_markdown("a\nb"), "<p>a<br>b</p>")

    def test_bold_and_italic(self) -> None:
        self.assertIn("<strong>重要</strong>", render_assistant_markdown("这是**重要**内容"))
        self.assertIn("<em>强调</em>", render_assistant_markdown("一个*强调*词"))

    def test_unordered_and_ordered_lists(self) -> None:
        html = render_assistant_markdown("- 甲\n- 乙\n- 丙")
        self.assertEqual(html, "<ul><li>甲</li><li>乙</li><li>丙</li></ul>")
        html = render_assistant_markdown("1. 第一\n2. 第二")
        self.assertEqual(html, "<ol><li>第一</li><li>第二</li></ol>")

    def test_separator(self) -> None:
        self.assertEqual(render_assistant_markdown("---"), "<hr>")

    def test_http_link_renders_as_anchor(self) -> None:
        html = render_assistant_markdown("看[这里](https://example.com/a)")
        self.assertIn('<a href="https://example.com/a"', html)
        self.assertIn('target="_blank"', html)
        self.assertIn("这里", html)

    def test_inline_code(self) -> None:
        html = render_assistant_markdown("试试 `play_track` 工具")
        self.assertIn("<code>play_track</code>", html)
        # emphasis must not leak into a code span
        self.assertNotIn("<strong>", render_assistant_markdown("`**x**`"))

    def test_cjk_and_mixed_content_keeps_visible_text(self) -> None:
        text = "《起风了（旧版）》**可以做**：\n\n- 试听\n- 播放\n\n详见 [Apple Music](https://music.apple.com)"
        html = render_assistant_markdown(text)
        self.assertIn("《起风了（旧版）》", html)
        self.assertIn("<strong>可以做</strong>", html)
        self.assertIn("<ul><li>试听</li><li>播放</li></ul>", html)
        self.assertIn('href="https://music.apple.com"', html)


class AssistantMarkdownSecurityTest(unittest.TestCase):
    """The wall: raw input never escapes, hostile markup never executes."""

    def test_raw_html_is_escaped_not_rendered(self) -> None:
        html = render_assistant_markdown("<script>alert(1)</script>")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_event_handler_attributes_are_escaped(self) -> None:
        # the whole construct is inert text; no tag, no live attribute
        html = render_assistant_markdown('<img src=x onerror="alert(1)">')
        self.assertNotIn("<img", html)
        self.assertTrue(html.startswith("<p>&lt;img"))
        self.assertIn("&quot;", html)  # the injected quotes stay quoted

    def test_javascript_link_renders_as_literal_text(self) -> None:
        html = render_assistant_markdown("[点我](javascript:alert(1))")
        self.assertNotIn("href", html)
        self.assertIn("[点我](javascript:alert(1))", html)  # visible, inert

    def test_data_link_renders_as_literal_text(self) -> None:
        html = render_assistant_markdown("[x](data:text/html,<script>)")
        self.assertNotIn("href", html)

    def test_link_url_with_quotes_cannot_break_the_attribute(self) -> None:
        html = render_assistant_markdown('[x](https://e.com/")')  # noqa: E501
        # the injected quote never terminates the href attribute; it stays an
        # escaped entity inside the attribute value
        self.assertNotIn('href="https://e.com/"', html)
        self.assertIn("&amp;quot;", html)

    def test_malicious_entity_content_stays_escaped(self) -> None:
        html = render_assistant_markdown("&lt;script&gt;x&lt;/script&gt;")
        self.assertNotIn("<script>", html)
        self.assertIn("&amp;lt;script&amp;gt;", html)

    def test_only_whitelisted_tags_appear(self) -> None:
        import re

        attack = (
            "<div class=x><iframe src=evil></iframe>"
            "**粗** *斜* [链](https://ok.com) `码` - 列表"
        )
        html = render_assistant_markdown(attack)
        tags = set(re.findall(r"</?([a-zA-Z0-9]+)", html))
        self.assertTrue(tags <= {"p", "br", "strong", "em", "ul", "ol", "li", "hr", "code", "a"})


if __name__ == "__main__":
    unittest.main()
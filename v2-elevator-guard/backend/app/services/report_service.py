"""PDF 报告。

用 reportlab 内置的 STSong-Light（CID 字体）显示中文，
不依赖系统字体文件，换机器也不用额外装字体。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

FONT = "STSong-Light"

RISK_COLORS = {
    "高风险": colors.HexColor("#c62828"),
    "中风险": colors.HexColor("#ef6c00"),
    "低风险": colors.HexColor("#2e7d32"),
}


def _styles() -> dict:
    base = getSampleStyleSheet()
    for name in ("Title", "Heading2", "BodyText"):
        base[name].fontName = FONT
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName=FONT, fontSize=19, leading=26
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#607d8b"),
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT,
            fontSize=13,
            leading=19,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#263238"),
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName=FONT, fontSize=10, leading=15.5
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8.5,
            leading=12.5,
            textColor=colors.HexColor("#546e7a"),
        ),
        "verdict": ParagraphStyle(
            "verdict",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=15,
            leading=21,
            alignment=TA_CENTER,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["BodyText"], fontName=FONT, fontSize=9, leading=13
        ),
        "cellhead": ParagraphStyle(
            "cellhead",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9,
            leading=13,
            textColor=colors.white,
        ),
    }


def _table(rows, styles, widths=None, header: bool = True) -> Table:
    table = Table(
        [
            [
                Paragraph(
                    str(cell),
                    styles["cellhead"] if (header and index == 0) else styles["cell"],
                )
                for cell in row
            ]
            for index, row in enumerate(rows)
        ],
        colWidths=widths,
        repeatRows=1 if header else 0,
    )
    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd8dc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
    table.setStyle(TableStyle(style))
    return table


class ReportService:
    def create(
        self,
        target: Path,
        item,
        behavior: dict | None = None,
        risk: dict | None = None,
        advice: dict | None = None,
        advice_variants: dict | None = None,
        overlay=None,
        keyframes: list[Path] | None = None,
        pose_engine_label: str = "",
    ) -> Path:
        behavior = behavior or {}
        risk = risk or {}
        advice = advice or {}
        advice_variants = advice_variants or {}
        keyframes = keyframes or []

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        pdfmetrics.registerFont(UnicodeCIDFont(FONT))
        styles = _styles()

        doc = SimpleDocTemplate(
            str(target),
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title=f"电梯危险行为分析报告 {item.id}",
        )
        story: list = []
        level = risk.get("level", item.risk_level or "低风险")
        color = RISK_COLORS.get(level, colors.HexColor("#546e7a"))

        story.append(Paragraph("电梯危险行为智能分析报告", styles["title"]))
        story.append(
            Paragraph(
                f"任务编号 {item.id} ｜ 生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["subtitle"],
            )
        )
        story.append(Spacer(1, 8))

        banner = Table(
            [
                [
                    Paragraph(
                        f"<font color='{color.hexval()}'><b>{level}</b></font>"
                        f"　｜　{behavior.get('label', item.behavior or '未分析')}",
                        styles["verdict"],
                    )
                ]
            ],
            colWidths=[doc.width],
        )
        banner.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 1, color),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafafa")),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        story.append(banner)
        story.append(Spacer(1, 10))

        warnings = (item.result or {}).get("warnings") or []
        if warnings:
            box = Table(
                [
                    [
                        Paragraph(
                            "<b>结论使用提示</b><br/>"
                            + "<br/>".join(f"· {text}" for text in warnings),
                            styles["body"],
                        )
                    ]
                ],
                colWidths=[doc.width],
            )
            box.setStyle(
                TableStyle(
                    [
                        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ef6c00")),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8e1")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            story.append(box)
            story.append(Spacer(1, 10))

        meta_rows = [
            ["项目", "内容"],
            ["源文件", item.filename],
            ["姿态模型", pose_engine_label or item.pose_engine or "—"],
            ["行为引擎", _engine_label(item.behavior_engine)],
            ["建议引擎", _engine_label(advice.get("engine") or item.advice_engine)],
            [
                "画面信息",
                f"{item.result.get('resolution', '—') if item.result else '—'} @ "
                f"{item.result.get('fps', '—') if item.result else '—'}fps，"
                f"时长 {item.duration_s or 0:.1f} 秒",
            ],
            [
                "采样情况",
                f"{item.frames_sampled or 0} 帧（总 {item.frames_total or 0} 帧），"
                f"最大同时人数 {item.people_max or 0}",
            ],
            ["关键点质量", f"{item.posture_quality or 0:.3f}"],
            [
                "可视化视频",
                (
                    f"{overlay.path.name}（{overlay.codec}，{overlay.frames} 帧）"
                    if overlay
                    else "未生成"
                ),
            ],
        ]
        story.append(_table(meta_rows, styles, widths=[30 * mm, doc.width - 30 * mm]))

        story.append(Paragraph("一、行为判定", styles["h2"]))
        story.append(
            Paragraph(
                f"判定结果：<b>{behavior.get('label', '—')}</b>"
                f"（{behavior.get('behavior', '—')}），"
                f"置信度 {float(behavior.get('confidence') or 0):.2f}。",
                styles["body"],
            )
        )
        if behavior.get("note"):
            story.append(Paragraph(f"说明：{behavior['note']}", styles["body"]))
        if behavior.get("reasoning"):
            story.append(Paragraph(f"判定理由：{behavior['reasoning']}", styles["body"]))
        evidence = behavior.get("evidence") or {}
        if evidence:
            lines = "；".join(f"{key} = {value}" for key, value in evidence.items())
            story.append(Paragraph(f"实测依据：{lines}", styles["body"]))

        story.append(Paragraph("二、风险评估", styles["h2"]))
        story.append(
            Paragraph(
                f"风险等级：<b>{level}</b>　风险分数：{float(risk.get('score') or 0):.2f}",
                styles["body"],
            )
        )
        if risk.get("reason"):
            story.append(Paragraph(risk["reason"], styles["body"]))
        factors = risk.get("factors") or []
        if factors:
            rows = [["风险因子", "取值"]]
            rows += [[str(f.get("name", "")), str(f.get("value", ""))] for f in factors]
            story.append(_table(rows, styles, widths=[60 * mm, doc.width - 60 * mm]))

        story.append(Paragraph("三、处置建议", styles["h2"]))
        story.append(
            Paragraph(
                f"<b>{advice.get('priority', '')} · {advice.get('title', '')}</b>",
                styles["body"],
            )
        )
        for action in advice.get("actions") or []:
            story.append(Paragraph(f"· {action}", styles["body"]))
        if advice.get("reasoning"):
            story.append(Paragraph(f"建议依据：{advice['reasoning']}", styles["body"]))
        if advice.get("note"):
            story.append(Paragraph(advice["note"], styles["small"]))
        if advice.get("disclaimer"):
            story.append(Paragraph(advice["disclaimer"], styles["small"]))

        references = advice.get("references") or []
        if references:
            rows = [["知识库条目", "标题", "来源", "相关度"]]
            for hit in references:
                rows.append(
                    [
                        hit.get("entry_id", ""),
                        hit.get("title", ""),
                        hit.get("source", ""),
                        f"{float(hit.get('score') or 0):.3f}",
                    ]
                )
            story.append(_table(rows, styles))
            for hit in references:
                story.append(
                    Paragraph(
                        f"<b>[{hit.get('entry_id')}] {hit.get('title')}</b><br/>"
                        f"{hit.get('snippet', '')}",
                        styles["small"],
                    )
                )

        if len(advice_variants) > 1:
            story.append(Paragraph("四、不同建议引擎对比", styles["h2"]))
            rows = [["引擎", "优先级", "标题", "首条建议", "耗时(ms)"]]
            for key, payload in advice_variants.items():
                actions = payload.get("actions") or []
                rows.append(
                    [
                        _engine_label(key),
                        payload.get("priority", ""),
                        payload.get("title", ""),
                        actions[0] if actions else "—",
                        str(payload.get("latency_ms", 0)),
                    ]
                )
            story.append(
                _table(
                    rows,
                    styles,
                    widths=[24 * mm, 18 * mm, 38 * mm, doc.width - 100 * mm, 20 * mm],
                )
            )

        if keyframes:
            story.append(PageBreak())
            story.append(Paragraph("五、图像证据", styles["h2"]))
            for path in keyframes:
                if not Path(path).exists():
                    continue
                try:
                    story.append(Paragraph(Path(path).name, styles["small"]))
                    story.append(
                        Image(
                            str(path),
                            width=doc.width,
                            height=doc.width * 0.5625,
                            kind="proportional",
                        )
                    )
                    story.append(Spacer(1, 8))
                except Exception:  # noqa: BLE001
                    continue

        story.append(Spacer(1, 10))
        story.append(Paragraph("附注与免责说明", styles["h2"]))
        notes = [
            "本报告由自动化系统生成：姿态估计模型提取关键点，行为引擎给出行为结论，"
            "Risk Agent 按确定性公式计算风险，建议引擎生成处置建议。",
            "知识库为通用电梯乘用安全语料，不替代正式国家标准文本；引用前请核对现行规程。",
            "行为规则阈值定义在 backend/app/engines/behavior/rules.yaml，可按现场条件调整。",
            "系统输出仅作为安全管理辅助依据，高风险结论须经人工确认后执行处置。",
        ]
        for note in notes:
            story.append(Paragraph(f"· {note}", styles["small"]))

        doc.build(story)
        logger.info("报告已生成：%s", target)
        return target


def _engine_label(key: str | None) -> str:
    from ..engines.advice import ADVICE_ENGINES
    from ..engines.behavior import BEHAVIOR_ENGINES
    from ..engines.pose import POSE_ENGINES

    if not key:
        return "—"
    for table in (POSE_ENGINES, BEHAVIOR_ENGINES, ADVICE_ENGINES):
        if key in table:
            return table[key].label
    return key

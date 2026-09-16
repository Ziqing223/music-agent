"""P10.8: Provider-facing projection of the 14 sealed P09 agent tools.

P09's tool registry carries validators, not JSON schemas; the provider needs machine
descriptions of what each tool accepts. This module is a pure projection -- it never
re-implements permission, validation, or execution: the SharedAgentService remains the
only execution boundary. Each schema mirrors the exact payload keys the P09 validators
enforce (reconciled from ``agent_tools.py``).
"""

from __future__ import annotations

from types import MappingProxyType

from music_agent.provider_contract import ProviderToolSchema

_TRACK_ID = {"type": "string", "description": "canonical track id (trk_ UUID)"}
_RUN_ID = {"type": "string", "description": "recommendation run id (rcm_ UUID)"}
_FEEDBACK_ID = {"type": "string", "description": "feedback observation id (fbk_ UUID)"}
_OPTIONAL_SOURCE = {
    "type": "string",
    "description": "source system (default: apple_music)",
}
_AWARE_ISO = {
    "type": "string",
    "description": "timezone-aware ISO 8601 instant, e.g. 2026-08-16T10:00:00+08:00",
}


def _schema(properties: dict, required: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


PROVIDER_TOOL_SCHEMAS: tuple[ProviderToolSchema, ...] = (
    ProviderToolSchema(
        "get_canonical_entity",
        "读取一个规范化实体（track/artist/album/playlist）的当前持久化状态。",
        _schema({"canonical_id": {"type": "string", "description": "entity id"}}, ("canonical_id",)),
    ),
    ProviderToolSchema(
        "query_track_preference",
        "查询一条轨道当前的偏好画像结论（不修改任何状态）。",
        _schema({"target_id": _TRACK_ID, "source_system": _OPTIONAL_SOURCE}, ("target_id",)),
    ),
    ProviderToolSchema(
        "list_recommendation_runs",
        "列出最近的推荐历史记录（时间倒序，最新在前；默认只返回最近 5 条，"
        "结果中 runs_total 为历史总数）。每首候选条目包含 name/artist_name "
        "显示名与 run_id/candidate_id/target_id，可定位「刚才推荐的」某首歌并用于反馈写入"
        "（当前批次优先读 get_active_context 的 active_batch 并用 get_recommendation_run "
        "读明细；active_batch 为 null 的老批次/兜底场景才用本工具找最近一批）。"
        "已在上文返回过的历史无需重复查询。",
        _schema(
            {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "返回的最近记录条数（默认 5）",
                }
            },
            (),
        ),
    ),
    ProviderToolSchema(
        "get_recommendation_run",
        "读取一次推荐运行的完整结果（含带 name/artist_name 显示名的候选条目与编码结果）。"
        "条目按推荐顺序返回：第 1 个条目即「第一首」，用户按「第 N 首」点播时取第 N 个条目"
        "（从 1 开始数）。",
        _schema({"run_id": _RUN_ID}, ("run_id",)),
    ),
    ProviderToolSchema(
        "generate_recommendation",
        "基于给定目标轨道生成一次推荐运行（会写入推荐历史）。"
        "用户要求「换一组/再来一批」时：可选 exclude_target_ids 填充上一批的全部 target_id，"
        "或设置 avoid_previous_runs 自动排除最近 5 批推荐过的轨道；"
        "两者只会让新一批与旧批次不同，不影响其它行为。"
        "调用未提供任何排除参数时本工具默认排除最近 5 批推荐过的轨道（短期去重，"
        "更早的历史推荐可重新出现）；显式传 avoid_previous_runs=false 才允许"
        "最近推荐过的轨道重新出现。"
        "生成结果为空时本工具返回错误、不写入任何推荐历史（空批次不会污染历史，"
        "也不会成为当前推荐批）；一次「换一组」只应调用本工具一次。"
        "target_ids 仅为目标范围（偏好引用），不是任意目录注入：本工具只会为带方向性"
        "直接偏好证据（喜欢/收藏/高评分 为正，不喜欢/低评分 为负）的轨道产生候选，"
        "没有这类证据的轨道会被直接跳过——尤其是新 discover 出来的全新目录曲目，"
        "它们通常没有任何偏好证据。"
        "当所有目标都没有正向直接偏好证据时，本工具通常不会返回空错误：会自动改用"
        "推断通道生成同一批推荐（结果带 channel=inferred_fallback 标注，"
        "与 generate_inferred_recommendation 的结果一致）；只有当推断通道也没有"
        "任何候选时才返回空错误。用户明确要「没听过的新歌」时仍请直接调用"
        "generate_inferred_recommendation 并传 min_fresh——本工具不会应用 Fresh 逻辑。"
        "任何新的推荐请求都应通过本工具生成新批次；"
        "历史批次仅用于用户明确指代（刚才的/上一批/第 N 首/换一首）的场景。",
        _schema(
            {
                "target_ids": {
                    "type": "array",
                    "items": _TRACK_ID,
                    "minItems": 1,
                    "description": "目标轨道 id 列表（目标范围/偏好引用：只有带方向性直接偏好证据的轨道才会成为候选）",
                },
                "limit": {"type": "integer", "minimum": 1, "description": "推荐数量上限"},
                "source_system": _OPTIONAL_SOURCE,
                "exclude_target_ids": {
                    "type": "array",
                    "items": _TRACK_ID,
                    "minItems": 1,
                    "description": "要排除的轨道 id（换一批时填上一批推荐的全部 target_id）",
                },
                "avoid_previous_runs": {
                    "type": "boolean",
                    "description": "true 时排除最近 5 批推荐过的轨道（短期去重，更早的历史推荐可重新出现）；缺省 true——调用未提供任何排除参数时按 true 处理；显式 false 允许最近推荐过的轨道重新出现",
                },
                "genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "方向过滤：只保留这些风格（如 [\"J-Pop\"]）的候选",
                },
            },
            ("target_ids", "limit"),
        ),
    ),
    ProviderToolSchema(
        "list_feedback_observations",
        "列出已记录的反馈观察（原始事实，不解释）。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "get_feedback_observation",
        "读取一条反馈观察记录。",
        _schema({"feedback_id": _FEEDBACK_ID}, ("feedback_id",)),
    ),
    ProviderToolSchema(
        "record_feedback",
        "记录一条用户明确反馈观察（如喜欢/不喜欢/完成/跳过）。"
        "target_id 与 (run_id + candidate_id) 二者只能提供其一；"
        "run_id 与 candidate_id 必须同时出现。"
        "观察时间 observed_at 由运行时权威生成，模型不可提供。",
        {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": "feedback kind，如 liked / disliked / completed / skipped",
                },
                "source_system": {"type": "string"},
                "source_path": {"type": "string"},
                "target_id": _TRACK_ID,
                "run_id": _RUN_ID,
                "candidate_id": {"type": "string", "description": "candidate id (cnd_ UUID)"},
                "attribution": {
                    "type": "object",
                    "description": "反馈归属证据",
                    "properties": {
                        "aspect_kind": {"type": "string", "enum": ["track", "artist", "album", "genre"]},
                        "aspect_id": {"type": "string"},
                        "relation": {"type": "string", "enum": ["attributed", "excluded"]},
                    },
                    "required": ["aspect_kind", "aspect_id", "relation"],
                    "additionalProperties": False,
                },
                "event_at": _AWARE_ISO,
                "source_event_id": {"type": "string"},
                "feedback_id": _FEEDBACK_ID,
            },
            "required": ["kind", "source_system", "source_path"],
            "additionalProperties": False,
            "oneOf": [
                {"required": ["target_id"]},
                {"required": ["run_id", "candidate_id"]},
            ],
        },
    ),
    ProviderToolSchema(
        "interpret_feedback",
        "对一条反馈观察做保守解释（跳过≠不喜欢，完成≠喜欢；无法断言时给出 NO_CLAIM）。",
        _schema({"feedback_id": _FEEDBACK_ID}, ("feedback_id",)),
    ),
    ProviderToolSchema(
        "list_learning_applications",
        "列出已应用的学习结论及其证据来源。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "get_learning_application",
        "读取一次学习应用结论。",
        _schema({"feedback_id": _FEEDBACK_ID}, ("feedback_id",)),
    ),
    ProviderToolSchema(
        "apply_learning",
        "把一条反馈证据提交为学习应用（作用于偏好画像的候选更新）；应用时间 applied_at 由运行时权威生成。",
        _schema({"feedback_id": _FEEDBACK_ID}, ("feedback_id",)),
    ),
    ProviderToolSchema(
        "get_agent_capabilities",
        "读取 Agent 能力清单（工具列表、写入能力 readiness、schema 版本）。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "execute_write_intent",
        "执行一个已存在的待定写入意图（受能力矩阵 gate 约束，未就绪时拒绝）。",
        _schema({"intent_id": {"type": "string", "description": "pending intent id (int_ UUID)"}}, ("intent_id",)),
    ),
    ProviderToolSchema(
        "play",
        "仅恢复/继续 Music.app 当前曲目的传输层播放（仅用户明确要求时调用；不做自动恢复）。"
        "本工具作用于当前播放上下文，绝不选择或切换曲目，也不接收曲目参数；"
        "用户点名播放某首指定歌曲时，绝不能代替 play_track —— 选曲播放必须使用 "
        "play_track(canonical_id)。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "pause",
        "暂停 Music.app 播放（用户要求，或耳机断开安全暂停）。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "next_track",
        "切到下一首。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "previous_track",
        "回到上一首。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "play_track",
        "选择并播放一首指定的规范化 Track（通过其 Apple Music persistent-ID 绑定解析，"
        "无绑定且无唯一库内等价曲目时失败关闭）。用户明确要求播放某首指定歌曲时，"
        "只要工具结果中已给出 canonical_id，就必须使用本工具选曲播放；"
        "通用 play 只负责恢复当前曲目，不能作为本工具的替代。",
        _schema({"canonical_id": _TRACK_ID}, ("canonical_id",)),
    ),
    ProviderToolSchema(
        "get_now_playing",
        "读取 Music.app 当前播放上下文（播放状态 + 当前曲目名/艺人/专辑）。"
        "结果仅为读取瞬间的快照，不含播放位置/剩余时长；播放命令之前的快照"
        "不能证明命令之后仍是同一曲目。需要在播放动作后核对实际播放内容时，"
        "请在动作之后再次调用本工具验证。"
        "结果还包含 context（agent_selected/own_queue/unknown：当前曲目是否属于最近"
        "Agent 推荐结果）与 agent_channel（library/preview/none：Agent 侧最近的播放/试听"
        "动作与曲目，仅本次运行有效、不持久化）；player_canonical_id 与 "
        "canonical_resolution 是实际 Music.app 当前曲目经既有严格 resolver 得到的只读"
        "规范身份与解析方式，无法唯一解析时均为 null；用户说「换一首」时先读这两项"
        "（context 与 agent_channel）判断所属上下文。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "get_active_context",
        "统一只读当前音乐交互上下文（视图层运行时信息，不持久化，不改变任何功能）："
        "channel = 本服务最近一次播放/试听动作（library/preview/none 及对应曲目 canonical_id）；"
        "referent_canonical_id = 最近一次明确曲目指代/成功动作的目标曲"
        "（播放/试听停止后仍保留，null=尚无；channel 为动作日志，与此不同）；"
        "preview_sounding = 试听是否真实在响（runner 真值，只读查询；channel 是动作日志，"
        "可能与 preview_sounding 不一致——判断「试听是否在播」只看本字段）；"
        "player = Music.app 当前曲目快照（与 get_now_playing 同源，不含播放位置；"
        "未接线时为 null，暂停/无播放时 state 为 stopped 且 persistent_id 为 null；"
        "canonical_id/canonical_resolution 是当前播放器曲目的只读、fail-closed 规范目标："
        "优先使用已有 persistent ID 绑定，否则仅接受唯一严格 playback-equivalent，"
        "无法唯一确认时两者为 null）；"
        "context = 当前曲目归属判断"
        "（agent_selected/own_queue/unknown）；active_batch = 当前推荐批次"
        "（run_id/source/产生时间/曲目数）：source=register 表示本服务刚交付的批次"
        "（运行时指针，权威），source=derived 表示无指针时从历史取的最新批次"
        "（新实例/跨进程的兜底）；条目明细（candidate/score/route/target 列表）不在此"
        "复制，需要时用 get_recommendation_run(run_id) 读取，从未推荐过时为 null。"
        "与 get_now_playing 的区别：本工具面向「判断整体上下文状态」，不是「核对具体播放"
        "内容」；判断用户「换一首」的对象归属时优先用本工具，核对刚播了什么继续用"
        "get_now_playing。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "generate_inferred_recommendation",
        "生成一次推荐运行：同一 source 内的直接偏好证据 + 基于真实 genre 传播的推断证据；"
        "候选池为目标 id 列表；直接正面为 known_positive，直接负面被排除，"
        "直接不足/未知 + 正向推断为 novel。"
        "本工具是 catalog 发现之后对「无直接证据曲目」的常规推荐接续：新的目录曲目通常"
        "没有任何直接偏好证据、generate_recommendation 无法为它们推荐，而本工具会用"
        "同一 source 内方向性证据传播出的 genre/artist 亲和度推断这些目标，"
        "并把带 catalog 绑定的目录曲目纳入候选。"
        "可选 genres：按用户方向（如 日系→J-Pop）只保留匹配风格的候选；"
        "可选 exclude_target_ids / avoid_previous_runs：换一批时排除上一批（精确）"
        "或最近 5 批内推荐过的轨道。"
        "调用未提供任何排除参数时本工具默认排除最近 5 批推荐过的轨道；"
        "显式传 avoid_previous_runs=false 才允许最近推荐过的轨道重新出现。"
        "可选 min_exploration（整数，0 ≤ 值 ≤ limit，默认 0）：最终批次中 best-effort "
        "至少保留该数量的「目录候选」（catalog 来源且已通过全部现有过滤与排名）——"
        "这是最终选择层的保留条目，不改变任何评分/资格/排序，合格目录候选不足时尽力而为；"
        "它不代表也不会保证「本次刚发现的新歌」。"
        "可选 min_fresh（整数，0 ≤ 值 ≤ limit，默认 0）：最终批次中 best-effort "
        "至少保留该数量的「本次刚发现的新歌」——身份由系统按本轮真实完成的 "
        "discover_catalog_tracks 的 promoted 结果自动判定（already_bound 永不算），"
        "模型提交的 any id 不参与判定；仅作用最终选择，不改评分/资格/排序，"
        "合格的新发现不足时尽力而为（少则少，绝不虚构）。"
        "传 min_fresh 时系统会自动保证探索下限不低于它（max 语义），"
        "一个合格条目可以同时满足两个下限，不会重复占名额。"
        "其中暂无任何偏好证据的新发现会以 score=0、无偏好依据的「探索发现」条目入批"
        "（只来自用户的明确探索请求）；介绍它们时不得声称与用户口味匹配，"
        "只能用「本次目录搜索的新发现，暂无偏好匹配证据」或同义措辞。"
        "普通推荐请求不要传 min_exploration 或 min_fresh；只有用户明确要求"
        "「新的/没听过的/库外的」且本轮已完成真实目录发现之后才值得传（≥1，不超过 limit）。"
        "生成结果的每个条目带 fresh_this_request 字段（true=本次请求真实新发现；"
        "false=已知目录/熟悉曲目），批次带 fresh_item_count（本批 true 的条数）；"
        "表述「新」只以这两个字段为准，不得自行比对名称/艺人/target_ids 推断。"
        "生成结果为空时本工具同样返回错误、不写入任何推荐历史；"
        "一次「换一组」只应调用本工具一次。"
        "任何新的推荐请求都应通过本工具生成新批次；"
        "历史批次仅用于用户明确指代（刚才的/上一批/第 N 首/换一首）的场景。",
        _schema(
            {
                "target_ids": {
                    "type": "array",
                    "items": _TRACK_ID,
                    "minItems": 1,
                    "description": "目标轨道 id 列表（无直接证据的轨道通过推断纳入候选池）",
                },
                "limit": {"type": "integer", "minimum": 1},
                "source_system": _OPTIONAL_SOURCE,
                "exclude_target_ids": {
                    "type": "array",
                    "items": _TRACK_ID,
                    "minItems": 1,
                    "description": "要排除的轨道 id（换一批时填上一批推荐的全部 target_id）",
                },
                "avoid_previous_runs": {
                    "type": "boolean",
                    "description": "true 时排除最近 5 批推荐过的轨道（短期去重，更早的历史推荐可重新出现）；缺省 true——调用未提供任何排除参数时按 true 处理；显式 false 允许最近推荐过的轨道重新出现",
                },
                "genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "方向过滤：只保留这些风格（如 [\"J-Pop\"]）的候选",
                },
                "min_exploration": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "最终批次中尽力保留的目录候选数量下限（默认 0 = 不强制探索；"
                    "仅作用于最终选择、不改评分/资格/排序；合格目录候选不足时尽力而为；"
                    "不保证本次刚发现的新歌）",
                },
                "min_fresh": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "最终批次中尽力保留的「本次刚发现的新歌」数量下限"
                    "（默认 0 = 不强制；Fresh 身份由系统按本轮 discover_catalog_tracks 的 "
                    "promoted 结果自动判定，模型提交的 id 不参与判定；仅作用于最终选择、"
                    "不改评分/资格/排序；合格新发现不足时尽力而为，绝不虚构；"
                    "零偏好证据的新发现以 score=0 入批，不得声称与用户口味匹配）",
                },
            },
            ("target_ids", "limit"),
        ),
    ),
    ProviderToolSchema(
        "preview_catalog_track",
        "播放一个规范化轨道的 30 秒音频预览：从其持久 itunes_store 绑定实时解析 iTunes "
        "previewUrl 并经由本机 afplay 边界播放（不修改任何本地库状态；"
        "无绑定或解析不到预览 URL 时失败关闭）。"
        "启动后立即返回 started，音频在后台播放约 30 秒；开始新的试听会自动停止上一首试听；"
        "用 stop_preview 停止当前试听。",
        _schema({"canonical_id": _TRACK_ID}, ("canonical_id",)),
    ),
    ProviderToolSchema(
        "stop_preview",
        "停止当前试听（Agent 通过 preview_catalog_track 发起的 afplay 后台播放；"
        "仅作用于试听边界，永远不会触碰 Music.app 播放）。"
        "无活动试听时幂等成功（stopped=false）。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "preview_batch",
        "把当前推荐批次逐首连续试听：「都放一遍/把这一批都试听一遍」类意图调用本工具"
        "（无参，队列由服务端按 active_batch 组装，无需提供曲目列表）。返回里的 session "
        "是权威会话状态快照（state=running/completed/failed/cancelled 及 position/total/"
        "current_name/skipped/failure_reason），started 只是 state==running 的派生标志："
        "启动后每首自动开始下一首（每首约 30 秒），无需为一首一首调用 preview_catalog_track"
        "也无需等待；停止连播用 stop_preview。回答用户时必须以上述 session.state 为准、"
        "绝不依据其他字段：state=running 时回答已开始连续试听（进度会陆续播报）；"
        "state=completed 时回答已全部试听完成；state=failed 时回答连播已中断"
        "（如实说明 failure_reason），绝不宣称已启动；state=cancelled 时回答已停止；"
        "started=false 时绝不宣称已启动。当前没有推荐批次、或批次曲目全部不可试听时"
        "失败关闭（preview_session_unavailable）。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "get_playback_context",
        "播放连续性只读合成视图（不修改任何状态）：channel=最近播放/试听动作、"
        "preview_sounding=试听是否真实在响、player=Music.app 当前曲目快照、"
        "session=连续试听会话进度（null=无连播进行）、suspended=被试听暂停过的正式播放"
        "留存（null=无）、referent_canonical_id=最近一次明确曲目指代/成功动作的目标曲"
        "（播放/试听停止后仍保留，null=尚无）。「试听到哪了」类进度查询用本工具。",
        _schema({}, ()),
    ),
    ProviderToolSchema(
        "add_catalog_to_library",
        "将一个已绑定 Catalog ID 的规范化轨道加入用户的 Apple Music 资料库，"
        "读取回执并核对 catalog-id 关系，将 persistent ID / ISRC 绑定到同一规范化轨道"
        "（真实外部写入，受写能力矩阵门控）。",
        _schema({"canonical_id": _TRACK_ID}, ("canonical_id",)),
    ),
    ProviderToolSchema(
        "open_in_apple_music",
        "在默认浏览器中打开一个规范化轨道的 Apple Music 官方页面：服务从轨道持久的 "
        "itunes_store 绑定实时解析 Apple 自己的 trackViewUrl（真实官方链接，"
        "绝不自行构造）并打开；结果 url 字段即真实链接，可直接展示给用户。"
        "本地资料库曲目（无目录绑定）会失败关闭（apple_music_open_unavailable）——"
        "此时如实说明没有链接，不得用搜索、试听或其他曲目替代。",
        _schema({"canonical_id": _TRACK_ID}, ("canonical_id",)),
    ),
    ProviderToolSchema(
        "discover_catalog_tracks",
        "在 Apple Music 曲库目录中按关键词搜索歌曲（只读目录 API，仅需开发者令牌），"
        "把新命中的歌曲去重/排除已知曲目后持久化为 catalog 候选（staging），"
        "再按权威 Catalog Artist/Album 身份解析关系并自动晋升（promote）为规范化轨道，"
        "返回 promoted/staged/skipped 明细（身份证据不足时保持 staged_blocked 并说明原因）。"
        "本工具仅兜底：库内同名/变体命中时不应急于动用目录；目录候选 preview_only 不得先于库内 library 选择。",
        _schema(
            {
                "term": {"type": "string", "description": "搜索关键词（歌曲名 / 艺人名等）"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "返回结果上限（默认 25）",
                },
            },
            ("term",),
        ),
    ),
    ProviderToolSchema(
        "search_library_tracks",
        "在 Music Agent 已知曲目记录中检索（只读、无网络、毫秒级）；结果混合 Apple Music Library、"
        "Apple Music Catalog/iTunes Store catalog 与其他 canonical records，不等同于用户的本地资料库。"
        "每条结果以 provenance 区分来源，以 bindings 显示 persistent/catalog identity，并附 playback.route。"
        "标题完全匹配或标题被完整包含在「标题 + 艺人」查询中时，保持标题相关性，并优先返回可正式播放的 "
        "Apple Music Library binding；catalog provenance 的记录不得称为用户资料库内容。"
        "用户点名播放歌名时：provenance.kind=apple_music_library 且 route=library 的结果用 play_track 正式播放；"
        "没有可信 Library 命中或结果不确定时才用 discover_catalog_tracks 查目录兜底。",
        _schema(
            {
                "term": {"type": "string", "description": "检索关键词（歌曲名 / 艺人名等）"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "返回匹配上限（默认 10）",
                },
            },
            ("term",),
        ),
    ),
    ProviderToolSchema(
        "query_catalog_discovery_state",
        "读取目录曲目的长期只读记忆（catalog_track_state，只查事实、不修改状态）："
        "given canonical_id 时返回该曲目的发现/推荐时间戳与计数，及派生标签"
        " never_recommended（从未被推荐过，recommendation_count=0）与"
        " previously_recommended（推荐过）；given term 时反查该搜索词曾产出过的"
        "已入库曲目（每首附该词出现次数与各自的推荐状态）。"
        "这是已见曲目的记忆/事实查询——不含排名、探索打分或资格判定，"
        "也不隐含对 Apple Music 目录新鲜度的任何结论；"
        "判断最新目录结果请用 discover_catalog_tracks（实时搜索）。",
        _schema(
            {
                "canonical_id": {
                    "type": "string",
                    "description": "canonical track id（trk_ UUID；与 term 二选一）",
                },
                "term": {
                    "type": "string",
                    "description": "discovery 搜索词（与 canonical_id 二选一，规范化匹配）",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "term 模式返回上限（默认 25）",
                },
            },
            (),
        ),
    ),
)

PROVIDER_TOOLS_BY_NAME: MappingProxyType = MappingProxyType(
    {schema.name: schema for schema in PROVIDER_TOOL_SCHEMAS}
)

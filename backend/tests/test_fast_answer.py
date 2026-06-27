from app.ai import AIClient


def test_fast_answer_structures_long_video_summary_and_filters_noise():
    timeline = [
        {"time": 53, "label": "3D 打印过程线索", "evidence": "三地打印一个房子里的所有东西"},
        {"time": 72, "label": "设计建模线索", "evidence": "进行所有的设计 制作装修"},
        {"time": 86, "label": "音频事件 3", "evidence": "不确"},
        {"time": 422, "label": "装修施工线索", "evidence": "就除了磁砖 墙面不是"},
        {"time": 602, "label": "3D 打印过程线索", "evidence": "成本是三地打印装修"},
        {"time": 616, "label": "音频事件 7", "evidence": "一公斤也得二三十"},
        {"time": 762, "label": "音频事件 8", "evidence": "就能实现一个无线接近传统家具的效果"},
    ]
    observations = [
        {
            "time": 55.67,
            "evidence_assessment": "画面显示的是一个聊天界面，没有3D打印设备，因此无法支持该音频描述。",
            "scene": "chat",
        },
        {
            "time": 603.49,
            "evidence_assessment": "画面显示一个蓝色的3D打印柜体，有明显的分层堆叠结构。",
            "scene": "printed cabinet",
        },
        {
            "time": 869.43,
            "evidence_assessment": "画面显示了家具设计的实物模型，有手部操作和设计图。",
            "scene": "furniture model",
        },
    ]

    answer = AIClient._fast_answer(
        "这个视频主要发生了什么？",
        {"timeline": timeline},
        observations,
        [],
    )

    assert "3D 打印装修实验" in answer["direct_answer"]
    assert "0:53" in answer["direct_answer"]
    assert "不确" not in answer["summary"]
    assert answer["sections"]
    assert [section["title"] for section in answer["sections"]] == ["内容脉络", "关键结论"]
    assert "关键画面证据" not in answer["summary"]
    assert answer["evidence_refs"] == []


def test_ai_overview_for_3d_printing_is_readable_and_filters_noisy_highlights():
    result = _printing_renovation_result()

    overview = AIClient.build_ai_overview(result)

    assert "视频主题" not in overview["summary"]
    assert "内容脉络：" not in overview["summary"]
    assert "3D 打印" in overview["summary"]
    assert overview["summary"].endswith("。")
    assert overview["bullets"]
    assert all(not item.startswith("内容脉络") for item in overview["bullets"])
    highlight_text = " ".join(f"{item.get('label', '')} {item.get('detail', '')}" for item in overview["highlights"])
    assert "就是给别人" not in highlight_text
    assert "不确" not in highlight_text
    assert any("成本" in item.get("label", "") or "成本" in item.get("detail", "") for item in overview["highlights"])


def test_3d_printing_followups_answer_specific_questions_without_asr_fragments():
    result = _printing_renovation_result()

    difficulty = AIClient._local_followup_answer("这个项目的主要难点是什么？", result)["answer"]
    cost = AIClient._local_followup_answer("成本和材料是怎么说的？", result)["answer"]
    effect = AIClient._local_followup_answer("最终效果怎么样？", result)["answer"]

    assert "材料" in difficulty
    assert "施工" in difficulty
    assert "成本" in difficulty
    assert "一公斤" in cost
    assert "耗材" in cost
    assert "传统家具" in effect or "层层堆叠" in effect
    joined = " ".join([difficulty, cost, effect])
    assert "就是给别人" not in joined
    assert "看看到底" not in joined
    assert "这个追问没有" not in joined


def test_fast_answer_summarizes_product_review_instead_of_evidence_dump():
    timeline = [
        {"time": 0.14, "label": "音频事件 1", "evidence": "当你点开这个视频说明你一定考虑过这个问题"},
        {"time": 2.94, "label": "音频事件 2", "evidence": "那就是这个Pockets 4P到底比Pockets要Pro多少"},
        {"time": 11.54, "label": "音频事件 3", "evidence": "那我猜你选它的原因是为了画质"},
        {"time": 14.14, "label": "音频事件 4", "evidence": "尤其它多了这颗等效60毫米的长焦"},
        {"time": 50.34, "label": "音频事件 5", "evidence": "Pockets 4P的外观其实都在我们预期之内的"},
        {"time": 162.99, "label": "音频事件 6", "evidence": "所以稳定效果你看一下"},
        {"time": 474.88, "label": "音频事件 7", "evidence": "我们测试下来发现它在ISO1600的时候动态范围是最好的"},
        {"time": 606.88, "label": "音频事件 8", "evidence": "那发热我们也测试了"},
    ]
    observations = [
        {"time": 477.48, "evidence_assessment": "画面显示昏暗室内场景，字幕为动态范围是最好的。", "scene": "low light test"},
    ]

    answer = AIClient._fast_answer("总结", {"timeline": timeline}, observations, [])

    assert "评测视频" in answer["direct_answer"]
    assert "Pocket 4P" in answer["direct_answer"]
    assert "0:00" in answer["direct_answer"]
    assert "音频给出事件线索" not in answer["direct_answer"]
    process = next(section for section in answer["sections"] if section["title"] == "内容脉络")
    joined = " ".join(process["items"])
    assert "购买疑问" in joined
    assert "画质测试" in joined
    assert "发热" in joined
    assert answer["evidence_refs"] == []


def test_followup_answers_upgrade_question_instead_of_repeating_summary():
    result = _pocket_review_result()

    answer = AIClient._local_followup_answer("总结一下比上一代有哪些升级", result)["answer"]

    assert "基于已分析内容" not in answer
    assert "不是全面换代" in answer
    assert "长焦" in answer
    assert "动态范围" in answer
    assert "其他很多基础体验" in answer
    assert "0:14" in answer
    assert "11:39" in answer


def test_followup_detailed_summary_reorganizes_video_by_question():
    result = _pocket_review_result()

    answer = AIClient._local_followup_answer("我需要详细总结", result)["answer"]

    assert "基于已分析内容" not in answer
    assert "详细来看" in answer
    assert "外观和硬件形态" in answer
    assert "动态范围测试" in answer
    assert "最终判断" in answer
    assert "7:55" in answer


def test_followup_plain_summary_does_not_fall_through_to_keyword_miss():
    result = _pocket_review_result()

    answer = AIClient._local_followup_answer("总结视频内容", result)["answer"]

    assert "这个追问没有" not in answer
    assert "详细来看" in answer
    assert "Pocket 4P" in answer
    assert "长焦" in answer
    assert "动态范围" in answer


def test_followup_buying_advice_recommends_generation_conditionally():
    result = _pocket_review_result()

    answer = AIClient._local_followup_answer("推荐买哪一代", result)["answer"]

    assert "这个追问没有" not in answer
    assert "更推荐 Pocket 4P" in answer
    assert "Pocket 4/普通版" in answer
    assert "重画质" in answer
    assert "重性价比" in answer


def test_answer_followup_uses_local_fast_path_for_common_questions(tmp_path):
    from test_workflow_mock import make_settings

    settings = make_settings(tmp_path)
    settings = settings.__class__(**{**settings.__dict__, "mock_mode": "false", "openai_api_key": "test-key"})
    ai = AIClient(settings)
    ai._client = object()

    answer = ai.answer_followup(question="推荐买哪一代", result=_pocket_review_result())

    assert "重画质" in answer["answer"]
    assert answer["coverage_note"] == ""


def test_time_specific_followup_is_not_overridden_by_summary_intent():
    result = _pocket_review_result()

    answer = AIClient._local_followup_answer("7分55秒左右在讲什么？", result)["answer"]

    assert "这个时间点附近" in answer
    assert "ISO1600" in answer or "动态范围" in answer
    assert "详细来看" not in answer


def test_local_followup_attaches_web_sources_without_replacing_video_answer():
    result = _pocket_review_result()
    web_context = {
        "enabled": True,
        "provider": "duckduckgo",
        "results": [{"title": "DJI Pocket 4P specs", "url": "https://example.test/specs", "snippet": "dual cameras"}],
    }

    answer = AIClient._attach_web_context(AIClient._local_followup_answer("推荐买哪一代", result), web_context)

    assert "更推荐 Pocket 4P" in answer["answer"]
    assert answer["web_sources"] == ["DJI Pocket 4P specs"]


def test_web_augmented_followup_answers_compound_summary_and_competitor_compare():
    result = _pocket_review_result()
    web_context = {
        "enabled": True,
        "provider": "duckduckgo",
        "results": [
            {
                "title": "Insta360 Luna Ultra - Flagship Dual-Lens Gimbal Camera",
                "snippet": "8K video, dual-lens system, 14 stops of dynamic range, Leica optics and low-light PureVideo mode.",
                "url": "https://example.test/luna",
            }
        ],
    }

    answer = AIClient._local_followup_answer(
        "总结这个视频内容，并对比跟insta360 luna哪个更好，区别是什么？",
        result,
        web_context=web_context,
    )

    assert "先说视频本身" in answer["answer"]
    assert "Insta360 Luna" in answer["answer"]
    assert "8K" in answer["answer"]
    assert "条件式" in answer["answer"] or "重实测" in answer["answer"]
    assert answer["web_sources"] == ["Insta360 Luna Ultra - Flagship Dual-Lens Gimbal Camera"]


def test_followup_knowledge_pack_retrieves_question_relevant_chunks_without_rules():
    result = _pocket_review_result()

    pack = AIClient._followup_knowledge_pack("低光动态范围表现怎么样", result)
    text = " ".join(chunk["text"] for chunk in pack["relevant_chunks"][:6])

    assert "动态范围" in text
    assert "ISO1600" in text or "17档" in text
    assert any(chunk.get("time_label") in {"7:55", "7:57", "7:58"} for chunk in pack["relevant_chunks"][:8])


def _pocket_review_result():
    return {
        "answer": {
            "direct_answer": "这是一期Pocket 4P评测视频，核心问题是它相比上一代/普通版是否值得多花钱升级。",
            "sections": [
                {"title": "关键结论", "items": ["Pocket 4P的画质和动态范围表现比预期好，长焦/双镜头是主要升级点。"]}
            ],
        },
        "transcript_segments": [
            {"start": 0.14, "end": 2.94, "text": "当你点开这个视频说明你一定考虑过这个问题"},
            {"start": 2.94, "end": 6.14, "text": "那就是这个Pockets 4P到底比Pockets要Pro多少"},
            {"start": 11.54, "end": 14.14, "text": "那我猜你选它的原因是为了画质"},
            {"start": 14.14, "end": 16.54, "text": "尤其它多了这颗等效60毫米的长焦"},
            {"start": 31.54, "end": 36.54, "text": "大江居然选择给Pockets 4P用了两块完全不一样的传感质"},
            {"start": 50.34, "end": 117.39, "text": "Pockets 4P的外观其实都在我们预期之内的"},
            {"start": 117.39, "end": 118.99, "text": "它就是一个双头的型态"},
            {"start": 128.99, "end": 130.79, "text": "还有Pockets 4上补光灯"},
            {"start": 159.59, "end": 162.99, "text": "所以电机扭力也调得比Pockets 4要大一些"},
            {"start": 265.19, "end": 266.39, "text": "它的画质到底怎么样呢"},
            {"start": 474.88, "end": 477.48, "text": "我们测试下来发现它在ISO1600的时候"},
            {"start": 477.48, "end": 478.48, "text": "动态范围是最好的"},
            {"start": 478.48, "end": 481.28, "text": "大概能够达到17档左右的动态范围"},
            {"start": 606.88, "end": 608.68, "text": "那发热我们也测试了"},
            {"start": 669.68, "end": 670.88, "text": "还有它的稳定性"},
            {"start": 698.90, "end": 700.70, "text": "我觉得Pocket 4P你买它多花的钱"},
            {"start": 700.70, "end": 702.90, "text": "其实就买在了一个画质还不错"},
            {"start": 707.10, "end": 708.10, "text": "然后还有这个动态范围"},
            {"start": 708.10, "end": 709.90, "text": "提升了很多的主摄镜头"},
            {"start": 709.90, "end": 711.50, "text": "别的东西跟Pocket 4真的一样"},
        ],
        "timeline": [],
        "frames": [],
    }


def _printing_renovation_result():
    timeline = [
        {"time": 31, "label": "成品效果线索", "evidence": "最后看成品效果：重点是柜体、家具或房间细节能否接近传统装修和家具的质感。"},
        {"time": 72, "label": "设计建模线索", "evidence": "先明确目标和方案：尝试用 3D 打印参与整屋装修/部件制作，并进行设计建模。"},
        {"time": 439, "label": "音频事件 5", "evidence": "就是给别人接着跳这个"},
        {"time": 478, "label": "3D 打印过程线索", "evidence": "随后进入打印和材料部分：关注打印设备、材料堆叠成型，以及可打印哪些装修/家具部件。"},
        {"time": 602, "label": "成本线索", "evidence": "后段讨论成本：包括材料单价、用量、整体预算，以及这种方案是否划算。"},
        {"time": 616, "label": "材料价格线索", "evidence": "一公斤也得二三十"},
        {"time": 762, "label": "成品效果线索", "evidence": "就能实现一个无限接近传统家具的效果"},
    ]
    answer = AIClient._fast_answer("总结当前视频内容", {"timeline": timeline}, [], [])
    return {
        "answer": answer,
        "audio_world_model": {"summary": "3D 打印装修实验，讨论设计、材料、成本和最终效果。"},
        "timeline": timeline,
        "transcript_segments": [
            {"start": 72, "end": 76, "text": "进行所有的设计 制作装修"},
            {"start": 478, "end": 482, "text": "打印材料层层堆叠"},
            {"start": 602, "end": 606, "text": "成本是三地打印装修"},
            {"start": 616, "end": 620, "text": "一公斤也得二三十"},
            {"start": 762, "end": 768, "text": "就能实现一个无限接近传统家具的效果"},
            {"start": 439, "end": 443, "text": "就是给别人接着跳这个"},
        ],
        "frames": [],
    }

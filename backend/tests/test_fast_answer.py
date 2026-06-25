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
    assert "不确" not in answer["summary"]
    assert answer["sections"]
    assert [section["title"] for section in answer["sections"]] == ["视频主题", "过程脉络", "关键画面证据", "结论"]
    evidence = next(section for section in answer["sections"] if section["title"] == "关键画面证据")
    assert all("无法支持" not in item for item in evidence["items"])

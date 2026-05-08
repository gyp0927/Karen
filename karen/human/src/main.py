"""
人类思维AI - 演示程序

运行方式：
    python main.py

这会启动一个交互式对话，你可以与具备人类思维特征的AI交流。
"""

import sys
import json
from datetime import datetime

# 添加src到路径
sys.path.insert(0, 'e:/人类的思维/src')

from mind.mind import Mind


def print_section(title: str, content: str, indent: int = 0):
    """格式化输出"""
    prefix = "  " * indent
    print(f"\n{prefix}{'='*60}")
    print(f"{prefix}{title}")
    print(f"{prefix}{'='*60}")
    if content:
        print(f"{prefix}{content}")


def print_emotion_state(mind):
    """打印情感状态"""
    state = mind.emotion.current
    print(f"  [情感] 效价:{state.valence:+.2f} 激活:{state.arousal:.2f} 控制:{state.dominance:.2f}")
    print(f"  [主导情绪] {mind.emotion.get_dominant_emotion()}")


def print_memory_state(mind):
    """打印记忆状态"""
    stats = mind.memory.get_memory_stats()
    print(f"  [记忆] 情景:{stats['episodic']} 语义:{stats['semantic']} 自传:{stats['autobiographical']}")
    wm = stats['working_memory']
    print(f"  [工作记忆] 内容:{wm['slots']} 负荷:{wm['load']:.2f}")


def run_interactive():
    """交互式对话模式"""
    print("=" * 70)
    print("  人类思维AI - 交互演示")
    print("=" * 70)
    print("\n这是一个模拟人类思维过程的AI系统。")
    print("它具备：情感、记忆、自我反思和创造性联想的能力。")
    print("\n你可以输入任何内容开始对话。")
    print("特殊命令：")
    print("  /state    - 查看AI的当前状态")
    print("  /reflect  - 让AI进行自我反思")
    print("  /memory   - 查看记忆统计")
    print("  /log      - 查看上一条思维过程日志")
    print("  /quit     - 退出")
    print("=" * 70)

    # 创建AI实例
    mind = Mind(name="Aurora")
    print(f"\n[系统] AI '{mind.name}' 已初始化")
    print(f"[系统] 初始情感状态: {mind.emotion.get_dominant_emotion()}")

    last_log = None

    while True:
        try:
            # 获取用户输入
            user_input = input("\n你: ").strip()

            if not user_input:
                continue

            # 处理特殊命令
            if user_input == "/quit":
                print("\n[系统] 保存会话状态并退出...")
                break

            elif user_input == "/state":
                print_section("当前状态", "")
                print_emotion_state(mind)
                print_memory_state(mind)
                print(f"  [对话轮次] {mind.total_turns}")
                print(f"  [身份叙事] {mind.get_identity()}")
                continue

            elif user_input == "/reflect":
                print_section("自我反思", mind.reflect())
                continue

            elif user_input == "/memory":
                print_section("记忆统计", "")
                stats = mind.memory.get_memory_stats()
                for key, value in stats.items():
                    if key != "working_memory":
                        print(f"  {key}: {value}")
                continue

            elif user_input == "/log":
                if last_log:
                    print_section("思维过程日志",
                                 json.dumps(last_log, indent=2, ensure_ascii=False))
                else:
                    print("  还没有思维日志。请先进行一次对话。")
                continue

            # 正常对话处理
            print("\n  [AI思考中...]")

            # 执行思维流程
            response, process_log = mind.think(user_input)
            last_log = process_log

            # 显示AI回应
            print(f"\n{mind.name}: {response}")

            # 简要显示内部状态变化
            print(f"\n  [内部状态] 情感:{mind.emotion.get_dominant_emotion()} | "
                  f"记忆:{mind.memory.get_memory_stats()['episodic']} | "
                  f"轮次:{mind.total_turns}")

        except KeyboardInterrupt:
            print("\n\n[系统] 收到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"\n[错误] {e}")

    # 退出时保存状态
    final_state = mind.save_state()
    print(f"\n[系统] 会话结束")
    print(f"[系统] 总计 {mind.total_turns} 轮对话")
    print(f"[系统] 最终情感: {mind.emotion.get_dominant_emotion()}")


def run_demo_scenario():
    """运行预设演示场景"""
    print("=" * 70)
    print("  人类思维AI - 预设演示")
    print("=" * 70)

    mind = Mind(name="Aurora")

    # 预设对话场景
    scenarios = [
        {
            "user": "Hello, how are you today?",
            "description": "简单问候 - 测试基础响应"
        },
        {
            "user": "I've been feeling really lost lately. I don't know what direction to take in life.",
            "description": "情感表达 - 测试共情回应"
        },
        {
            "user": "What do you think about consciousness? Can AI truly be conscious?",
            "description": "哲学问题 - 测试深度思考"
        },
        {
            "user": "I want to write a poem about time, but not the cliché 'time is like a river'. Any ideas?",
            "description": "创造性任务 - 测试联想和隐喻"
        },
        {
            "user": "You were wrong about what I meant earlier.",
            "description": "冲突 - 测试错误承认"
        },
        {
            "user": "Do you remember our first conversation?",
            "description": "记忆测试 - 测试情景记忆"
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"  场景 {i}/{len(scenarios)}: {scenario['description']}")
        print(f"{'='*70}")

        print(f"\n用户: {scenario['user']}")

        # 显示处理前的状态
        print(f"\n  [处理前状态]")
        print_emotion_state(mind)

        # 处理
        response, log = mind.think(scenario['user'])

        # 显示回应
        print(f"\n{mind.name}: {response}")

        # 显示处理后的状态
        print(f"\n  [处理后状态]")
        print_emotion_state(mind)
        print_memory_state(mind)

        # 显示思维过程摘要
        print(f"\n  [思维过程摘要]")
        if 'phases' in log:
            for phase_name, phase_data in log['phases'].items():
                if phase_name == 'perception':
                    print(f"    - 感知: 检测到情绪 {phase_data.get('detected_emotion', {}).get('dominant_emotion', 'neutral')}, "
                          f"紧急度 {phase_data.get('urgency', 0):.2f}")
                elif phase_name == 'decision':
                    print(f"    - 决策: 最佳策略 '{phase_data.get('best_option', 'unknown')}', "
                          f"信心 {phase_data.get('confidence', 0):.2f}")
                elif phase_name == 'metacognition_output':
                    print(f"    - 元认知: 确定性检查 {phase_data.get('certainty_check', {}).get('assessment', 'unknown')}")

        input("\n按回车继续...")

    print(f"\n{'='*70}")
    print("  演示结束")
    print(f"{'='*70}")
    print(f"\n最终状态:")
    print(f"  对话轮次: {mind.total_turns}")
    print(f"  情感历史长度: {len(mind.emotion.history)}")
    print(f"  记忆数量: {mind.memory.get_memory_stats()['episodic']}")


def run_quick_test():
    """快速测试 - 非交互模式"""
    print("=" * 70)
    print("  快速测试")
    print("=" * 70)

    mind = Mind(name="Aurora")

    test_inputs = [
        "Hello!",
        "I'm feeling a bit sad today.",
        "What is the meaning of life?",
    ]

    for user_input in test_inputs:
        print(f"\n用户: {user_input}")
        response, log = mind.think(user_input)
        print(f"AI: {response}")
        print(f"  [情感: {mind.emotion.get_dominant_emotion()}, "
              f"记忆: {mind.memory.get_memory_stats()['episodic']}, "
              f"轮次: {mind.total_turns}]")

    print("\n测试完成!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="人类思维AI演示")
    parser.add_argument("--mode", choices=["interactive", "demo", "test"],
                       default="interactive",
                       help="运行模式: interactive(交互式), demo(演示), test(快速测试)")

    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive()
    elif args.mode == "demo":
        run_demo_scenario()
    elif args.mode == "test":
        run_quick_test()

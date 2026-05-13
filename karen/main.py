import asyncio
import logging

from agents.nodes import create_agents
from interface.human_interface import HumanInterface
from state.manager import SessionManager

# 配置控制台日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main():
    print("=" * 50)
    print("凯伦")
    print("=" * 50)
    print("\nInitializing agents...")

    # 创建 Agents
    coordinator, researcher, responder, reviewer = create_agents()
    print("Agents created successfully!")

    # 初始化消息管理器
    msg_manager = SessionManager()

    # 创建用户接口（启用协调模式 + 审查）
    interface = HumanInterface(
        message_manager=msg_manager,
        coordinator=coordinator,
        researcher=researcher,
        responder=responder,
        reviewer=reviewer,
        fast_mode=True,
        review=False,
        review_language="zh",
    )

    print("\nSystem ready! Type 'exit' to quit.")
    print("Commands:")
    print("  /review    - Toggle review on/off")
    print("  /fast      - Toggle fast mode")
    print("  /clear     - Clear current conversation")
    print("  /history   - Show conversation history\n")

    # 对话循环
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # 处理命令
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break
            elif user_input.lower() == "/review":
                interface.review = not interface.review
                print(f"\nReview {'enabled' if interface.review else 'disabled'}\n")
                continue
            elif user_input.lower() == "/fast":
                interface.fast_mode = not interface.fast_mode
                mode = "Fast" if interface.fast_mode else "Coordination"
                print(f"\nSwitched to {mode} mode\n")
                continue
            elif user_input.lower() == "/clear":
                msg_manager.clear()
                print("\nConversation cleared.\n")
                continue
            elif user_input.lower() == "/history":
                history = msg_manager.get_messages()
                print(f"\n--- History ({len(history)} messages) ---")
                for msg in history:
                    sender = getattr(msg, "name", "unknown")
                    print(f"[{sender}]: {msg.content[:100]}...")
                print("---\n")
                continue

            response = await interface.send_message(user_input)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.exception("Error in conversation loop")
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())

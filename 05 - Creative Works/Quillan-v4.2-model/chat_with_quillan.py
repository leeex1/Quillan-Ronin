#!/usr/bin/env python3
"""
Interactive Chat Interface for Quillan-Ronin v5.3.1
Talk to your trained multimodal AI!
"""

import torch
import torch.nn.functional as F
from train_full_multimodal import QuillanRoninV5_3, Config, SimpleTokenizer
from data_loader import QuillanDataset
import sys

class QuillanChat:
    def __init__(self):
        self.model = None
        self.cfg = None
        self.tokenizer = None
        self.device = torch.device('cpu')
        self.conversation_history = []

    def load_model(self):
        """Load the trained model"""
        print("🤖 Loading Quillan-Ronin v5.3.1...")
        print("-" * 50)

        try:
            # Load model configuration
            self.cfg = Config()
            self.model = QuillanRoninV5_3(self.cfg)

            # Load checkpoint
            checkpoint = torch.load("best_multimodal_quillan.pt", map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Move to device
            self.model = self.model.to(self.device)
            self.cfg.device = self.device

            print("✅ Model loaded successfully!")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

        try:
            # Setup tokenizer
            print("🏗️ Setting up tokenizer...")
            dataset = QuillanDataset()
            self.tokenizer = SimpleTokenizer(vocab_size=1000)
            all_texts = [s['text'] for s in dataset.samples]
            self.tokenizer.train(all_texts)

            print(f"✅ Tokenizer ready with {len(self.tokenizer.char_to_idx)} tokens!")

        except Exception as e:
            print(f"❌ Failed to setup tokenizer: {e}")
            return False

        return True

    def generate_response(self, user_input, max_length=100):
        """Generate a response to user input"""
        try:
            # Encode user input
            prompt_tokens = self.tokenizer.encode(user_input, max_length=50)
            generated_tokens = prompt_tokens.copy()

            # Create multimodal inputs (dummy for chat)
            batch_size = 1
            dummy_image = torch.randn(batch_size, 3, 256, 256, device=self.device)
            dummy_audio = torch.randn(batch_size, 1, 2048, device=self.device)
            dummy_video = torch.randn(batch_size, 3, 8, 32, 32, device=self.device)

            self.model.eval()
            with torch.no_grad():
                for _ in range(max_length):
                    input_text = torch.tensor([generated_tokens], device=self.device)
                    outputs = self.model(input_text, dummy_image, dummy_audio, dummy_video)

                    # Get next token logits
                    text_logits = outputs['text'][0, -1, :]

                    # Strong bias against pad/unk tokens for chat
                    text_logits[0] = -1000  # Pad token
                    text_logits[1] = -500   # Unknown token

                    # Temperature for creativity
                    text_logits = text_logits / 0.9

                    probabilities = F.softmax(text_logits, dim=-1)

                    # Sample next token
                    next_token = torch.multinomial(probabilities, 1).item()

                    # Stop conditions
                    if next_token in [0, 1] and len(generated_tokens) > len(prompt_tokens) + 5:
                        break
                    if len(generated_tokens) >= max_length + len(prompt_tokens):
                        break

                    generated_tokens.append(next_token)

            # Decode response
            response = ""
            for token in generated_tokens[len(prompt_tokens):]:  # Skip input tokens
                if token in self.tokenizer.idx_to_char:
                    response += self.tokenizer.idx_to_char[token]

            # Clean up response
            response = response.strip()
            if not response:
                response = "..."

            return response

        except Exception as e:
            return f"Error generating response: {e}"

    def start_chat(self):
        """Start interactive chat session"""
        if not self.load_model():
            print("❌ Could not initialize chat. Check model file and try again.")
            return

        print("\n" + "="*60)
        print("🎭 WELCOME TO QUILLAN-RONIN CHAT v5.3.1")
        print("="*60)
        print("🤖 I'm your multimodal AI assistant!")
        print("💬 Type your messages and I'll respond.")
        print("❌ Type 'quit' or 'exit' to end the conversation.")
        print("🔄 Type 'clear' to clear conversation history.")
        print("-" * 60)

        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Quillan: Goodbye! It was great chatting with you!")
                    print("📊 Conversation ended. Have a great day!")
                    break

                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🧹 Conversation history cleared!")
                    continue

                if user_input.lower() == 'help':
                    print("\n📚 Available commands:")
                    print("  'quit'/'exit' - End conversation")
                    print("  'clear' - Clear history")
                    print("  'help' - Show this help")
                    continue

                # Add to history
                self.conversation_history.append(f"You: {user_input}")

                # Generate response
                print("🤖 Quillan: Thinking...", end="", flush=True)
                response = self.generate_response(user_input)
                print(f"\r🤖 Quillan: {response}")

                # Add response to history
                self.conversation_history.append(f"Quillan: {response}")

                # Show conversation stats
                if len(self.conversation_history) % 4 == 0:
                    print(f"📊 Conversation turns: {len(self.conversation_history)//2}")

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in chat: {e}")
                continue

    def show_conversation_summary(self):
        """Show summary of the conversation"""
        if not self.conversation_history:
            print("📝 No conversation history.")
            return

        print(f"\n📊 Conversation Summary:")
        print(f"   Total messages: {len(self.conversation_history)}")
        print(f"   Conversation turns: {len(self.conversation_history)//2}")

        if len(self.conversation_history) <= 10:
            print("\n💬 Full conversation:")
            for i, msg in enumerate(self.conversation_history, 1):
                print(f"   {i}. {msg}")
        else:
            print("\n💬 Recent conversation:")
            for i, msg in enumerate(self.conversation_history[-6:], len(self.conversation_history)-5):
                print(f"   {i}. {msg}")

def main():
    """Main chat function"""
    print("🚀 Starting Quillan-Ronin Interactive Chat")
    print("This will load the trained model and allow you to chat with it.")
    print("Make sure 'best_multimodal_quillan.pt' exists in the current directory.")
    print()

    chat = QuillanChat()
    chat.start_chat()
    chat.show_conversation_summary()

if __name__ == "__main__":
    main()

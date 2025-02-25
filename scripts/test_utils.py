import unittest

from utils import remove_emojis, remove_links_and_tags


class TestUtils(unittest.TestCase):
    def test_remove_links_and_tags(self):
        text = "Check this out http://example.com @user1"
        expected = "Check this out"
        result = remove_links_and_tags(text)
        if result != expected:
            print("Unexpected behavior:")
            print("Input:", text)
            print("Expected:", expected)
            print("Result:", result)
        self.assertEqual(result, expected)

        text = "Visit www.example.com for more info @user2"
        expected = "Visit for more info"
        result = remove_links_and_tags(text)
        if result != expected:
            print("Unexpected behavior:")
            print("Input:", text)
            print("Expected:", expected)
            print("Result:", result)
        self.assertEqual(result, expected)

        text = "No links or tags here!"
        expected = "No links or tags here!"
        result = remove_links_and_tags(text)
        if result != expected:
            print("Unexpected behavior:")
            print("Input:", text)
            print("Expected:", expected)
            print("Result:", result)
        self.assertEqual(result, expected)

    def test_remove_emojis(self):
        self.assertEqual(remove_emojis("Hello 😊"), "Hello")
        self.assertEqual(remove_emojis("No emojis here!"), "No emojis here!")
        self.assertEqual(remove_emojis("Multiple emojis 😂😂😂"), "Multiple emojis")
        self.assertEqual(
            remove_emojis("Mixed content: text, emojis 😊, and symbols #!"),
            "Mixed content: text, emojis , and symbols #!",
        )
        self.assertEqual(remove_emojis("Flags 🇺🇸🇨🇦"), "Flags")
        self.assertEqual(
            remove_emojis("十二月十五，美國寘無人潜航器於漲海，而中國海軍穫之。"),
            "十二月十五，美國寘無人潜航器於漲海，而中國海軍穫之。",
        )


if __name__ == "__main__":
    unittest.main()

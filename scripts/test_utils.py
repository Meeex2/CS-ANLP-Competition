import unittest

from utils import remove_links_and_tags


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


if __name__ == "__main__":
    unittest.main()

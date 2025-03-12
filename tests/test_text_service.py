import unittest
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.text_service import extract_participants, parse_timestamp

class TestTextService(unittest.TestCase):
    
    def test_extract_participants_pattern1(self):
        # Test Pattern 1: Name (Role): Text
        transcript = """
        Alice (Manager): Welcome to the meeting everyone.
        Bob (Developer): Thanks for having us.
        Charlie (Designer): I'm excited to discuss the project.
        """
        
        participants = extract_participants(transcript)
        self.assertEqual(set(participants), {"Alice", "Bob", "Charlie"})
    
    def test_extract_participants_pattern2(self):
        # Test Pattern 2: Name: Text
        transcript = """
        Alice: Hello everyone, let's start the meeting.
        Bob: Sure, I've prepared some slides.
        Charlie: I have some design ideas to share.
        """
        
        participants = extract_participants(transcript)
        self.assertEqual(set(participants), {"Alice", "Bob", "Charlie"})
    
    def test_extract_participants_pattern3(self):
        # Test Pattern 3: Speaker X
        transcript = """
        [00:01:15] Speaker 1: Let's begin the discussion.
        [00:01:30] Speaker 2: I agree with the approach.
        [00:01:45] Speaker 3: I have some concerns about the timeline.
        """
        
        participants = extract_participants(transcript)
        self.assertEqual(set(participants), {"Speaker 1", "Speaker 2", "Speaker 3"})
    
    def test_extract_participants_mixed(self):
        # Test mixed patterns
        transcript = """
        Alice (Product Manager): Welcome everyone.
        Bob: Thanks Alice.
        [00:02:15] Speaker 3: I'd like to add something.
        Charlie (Designer): Sure, go ahead.
        """
        
        participants = extract_participants(transcript)
        self.assertEqual(set(participants), {"Alice", "Bob", "Charlie", "Speaker 3"})
    
    def test_parse_timestamp_hhmmss(self):
        # Test HH:MM:SS format
        timestamp = "01:23:45"
        seconds = parse_timestamp(timestamp)
        self.assertEqual(seconds, 5025.0)  # 1*3600 + 23*60 + 45
    
    def test_parse_timestamp_mmss(self):
        # Test MM:SS format
        timestamp = "12:34"
        seconds = parse_timestamp(timestamp)
        self.assertEqual(seconds, 754.0)  # 12*60 + 34
    
    def test_parse_timestamp_with_brackets(self):
        # Test timestamp with brackets
        timestamp = "[00:45:30]"
        seconds = parse_timestamp(timestamp)
        self.assertEqual(seconds, 2730.0)  # 45*60 + 30
    
    def test_parse_timestamp_invalid(self):
        # Test invalid timestamp
        timestamp = "invalid"
        seconds = parse_timestamp(timestamp)
        self.assertIsNone(seconds)

if __name__ == '__main__':
    unittest.main()
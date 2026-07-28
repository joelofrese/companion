import unittest

from onboard.bringup_check import check_command, check_module, check_udp_port


class BringupCheckTests(unittest.TestCase):
    def test_reports_known_module_and_missing_command(self):
        self.assertTrue(check_module("json").passed)
        self.assertFalse(check_command("companion-command-that-does-not-exist").passed)

    def test_can_check_an_ephemeral_udp_port(self):
        result = check_udp_port(0, host="127.0.0.1")
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()

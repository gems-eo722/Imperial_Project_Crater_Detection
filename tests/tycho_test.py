import os
import unittest

from tycho_cdm import tycho


class TychoTest(unittest.TestCase):
    def test_dir_is_empty(self):
        here = os.path.abspath(os.path.dirname(__file__))
        new_folder_path = os.path.join(here, 'new_folder')
        os.makedirs(new_folder_path)
        self.assertEqual(tycho.dir_is_empty(new_folder_path), True)
        os.removedirs(new_folder_path)

    def test_dir_is_not_empty(self):
        here = os.path.abspath(os.path.dirname(__file__))
        new_folder_path = os.path.join(here, 'new_folder')
        os.makedirs(new_folder_path)
        new_file_path = os.path.join(new_folder_path, 'new_file.txt')
        with open(new_file_path, 'w'):
            self.assertEqual(tycho.dir_is_empty(new_folder_path), False)
        os.remove(new_file_path)
        os.removedirs(new_folder_path)

    def test_dir_is_empty_file(self):
        here = os.path.abspath(os.path.dirname(__file__))
        new_file_path = os.path.join(here, 'new_file.txt')
        with open(new_file_path, 'w'):
            self.assertRaises(NotADirectoryError, lambda: tycho.dir_is_empty(new_file_path))
        os.remove(new_file_path)

    def test_parser_rejects_empty_args(self):
        parser = tycho.make_parser()
        self.assertRaises(SystemExit, lambda: parser.parse_args([]))

    def test_parser_rejects_only_input(self):
        parser = tycho.make_parser()
        self.assertRaises(SystemExit, lambda: parser.parse_args(["-i", "input_folder"]))

    def test_parser_rejects_only_input_output(self):
        parser = tycho.make_parser()
        self.assertRaises(SystemExit, lambda: parser.parse_args(["-i", "input_folder", "-o", "output_folder"]))

    def test_parser_accepts_input_output_planet_name(self):
        parser = tycho.make_parser()
        args = parser.parse_args(["-i", "inp", "-o", "out", "-p", "mars"])
        self.assertEqual(args.input_folder, "inp")
        self.assertEqual(args.output_folder, "out")
        self.assertEqual(args.planet_name, "mars")

    def test_check_arguments(self):
        pass


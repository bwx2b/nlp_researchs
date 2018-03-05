#!/usr/env/bin python
#coding=utf-8

import ConfigParser as configparser

class Parser(object):
    def __init__(self, config_file):
        self._configParser = configparser.ConfigParser()
        self._configParser.read(config_file)

    def read_sections(self):
        """
        获取配置文件所有的节点
        :return:
        """
        return self._configParser.sections()

    def read_items(self, section):
        """
        获取指定节点下所有的键值对
        :return:
        """
        return self._configParser.items(section)

    def read_option(self, section):
        """
        获取指定节点下的所有键
        :param section:
        :return:
        """
        return self.options(section)

    def get(self, section, option):
        """
        获取指定节点指定键的值
        :param section:
        :param option:
        :return:
        """
        return self._configParser.get(section, option)


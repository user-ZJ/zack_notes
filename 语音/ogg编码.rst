ogg编码
==============================================

OGG 是一种 开源、免费的多媒体容器格式，主要用于存储音频、视频或其他数据。它支持多种编解码器，其中最常见的是 Vorbis 音频编码（替代 MP3 的高效压缩格式）和 Theora 视频编码。OGG 格式以其 高压缩率、无损元数据支持 和 无专利限制 的特点，常用于流媒体、音频 / 视频文件存储及多媒体项目。

OGG 头部（Ogg Page）结构概述
-----------------------------------------

+-------------------------+------------+--------------------------------------------------------------------------------------------------+
|          字段           | 长度(字节) |                                               描述                                               |
+=========================+============+==================================================================================================+
| 同步字（Sync Word）     | 4          | 固定值 0x4F676753（即 "OggS"，大端序），标识 OGG 格式                                            |
+-------------------------+------------+--------------------------------------------------------------------------------------------------+
| 版本（Version）         | 1          | 目前固定为 0                                                                                     |
+-------------------------+------------+--------------------------------------------------------------------------------------------------+
| 标志位（Flags）         | 1          | 包含页面类型（如是否为第一个 Page、是否为最后一个 Page 等）;0x02为第一个Page，0x04为最后一个Page |
+-------------------------+------------+--------------------------------------------------------------------------------------------------+
| 绝对时间戳（Timestamp） | 8          | 小端序无符号整数，单位为 纳秒（音频 / 视频的时间基准）                                           |
+-------------------------+------------+--------------------------------------------------------------------------------------------------+
| 页码（Page Sequence）   | 4          | 小端序无符号整数，页面编号，从 0 开始递增                                                        |
+-------------------------+------------+--------------------------------------------------------------------------------------------------+
| 页码（Page Sequence）   | 4          | 小端序无符号整数，页面编号，从 0 开始递增                                                        |
+-------------------------+------------+--------------------------------------------------------------------------------------------------+
| CRC校验码               | 4          | 小端序无符号整数，页面编号，从 0 开始递增                                                        |
+-------------------------+------------+--------------------------------------------------------------------------------------------------+
| 段数（Segment Count）   | 1          | 该 Page 包含的音频 / 视频段数量（每个段长度由后续字节描述）                                      |
+-------------------------+------------+--------------------------------------------------------------------------------------------------+


header解析
-----------------------------------------
.. code-block:: cpp

    struct OggHeader {
        char sync[4]; // 同步字
        uint8_t version; // 版本
        uint8_t flags; // 标志位
        uint64_t granule_position; // 绝对时间戳
        uint32_t stream_serial_number; // 页码
        uint32_t page_sequence_number; // 页码
        uint32_t checksum; // CRC校验码
        uint8_t segments_count; // 段数
        unsigned char segmet_table[]; // 段长度数组
    };

    int ogg_header_parse(const char *header, int header_len, OggHeader *ogg_header)
    {
        if (header_len < 27) {
            return -1;
        }
        if (memcmp(header, "OggS", 4) != 0) {
            return -1;
        }
        ogg_header->version = header[4];
        ogg_header->flags = header[5];
        ogg_header->granule_position = *(uint64_t *)(header + 6);
        ogg_header->stream_serial_number = *(uint32_t *)(header + 14);
        ogg_header->page_sequence_number = *(uint32_t *)(header + 18);
        ogg_header->checksum = *(uint32_t *)(header + 22);
        ogg_header->segments_count = header[26];
        return 0;
    }


.. code-block:: python

    import struct

    class OggHeader:
        def __init__(self):
            self.sync = b''
            self.version = 0
            self.flags = 0
            self.granule_position = 0
            self.stream_serial_number = 0
            self.page_sequence_number = 0
            self.checksum = 0
            self.segments_count = 0
            self.segment_table = []

        def parse(self, filebytes):
            if len(filebytes) < 27:
                return -1
            if filebytes[:4] != b'OggS':
                return -1
            self.sync = filebytes[:4]
            self.version, self.flags, self.granule_position, \
                self.stream_serial_number, self.page_sequence_number, \
                self.checksum, self.segments_count = struct.unpack('<B B Q I I I B', filebytes[4:27])
            flags_table = {
                0x02: 'first_page',
                0x04: 'last_page',
                0x01: 'continued_page',
                0x08: 'b_o_s',
                0x10: 'e_o_s'
            }
            is_first_page = self.flags == 0x02
            for i in range(self.segments_count):
                size = struct.unpack('<B', filebytes[27 + i])
                self.segment_table.append(size)
            page_sum_size = 0
            for i in range(self.segments_count):
                page_sum_size += self.segment_table[i]
            page_content = filebytes[27 + self.segments_count: 27 + self.segments_count + page_sum_size]
            return page_content,is_first_page
            

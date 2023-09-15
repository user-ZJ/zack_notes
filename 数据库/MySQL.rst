MySQL
=================

数据类型
---------------------
MySQL支持以下数据类型：

1. 整数类型（Integer Types）：TINYINT、SMALLINT、MEDIUMINT、INT、BIGINT
2. 浮点数类型（Floating-Point Types）：FLOAT、DOUBLE、DECIMAL
3. 日期和时间类型（Date and Time Types）：DATE、TIME、DATETIME、TIMESTAMP、YEAR
4. 字符串类型（String Types）：CHAR、VARCHAR、TINYTEXT、TEXT、MEDIUMTEXT、LONGTEXT
5. 二进制数据类型（Binary Data Types）：BINARY、VARBINARY、TINYBLOB、BLOB、MEDIUMBLOB、LONGBLOB
6. 枚举类型（Enumeration Types）
7. 集合类型（Set Types）
8. 几何类型（Geometry Types）
9. JSON 类型

整数类型（Integer Types）
`````````````````````````````````````
* TINYINT: 1 字节（范围：-128 到 127，无符号类型的范围：0 到 255）
* SMALLINT: 2 字节（范围：-32768 到 32767，无符号类型的范围：0 到 65535）
* MEDIUMINT: 3 字节（范围：-8388608 到 8388607，无符号类型的范围：0 到 16777215）
* INT: 4 字节（范围：-2147483648 到 2147483647，无符号类型的范围：0 到 4294967295）
* BIGINT: 8 字节（范围：-9223372036854775808 到 9223372036854775807，无符号类型的范围：0 到 18446744073709551615）

浮点数类型（Floating-Point Types）
`````````````````````````````````````````````
* FLOAT: 4 字节单精度浮点数（范围：-3.402823466E+38 到 -1.175494351E-38、0 和 1.175494351E-38 到 3.402823466E+38）
* DOUBLE: 8 字节双精度浮点数（范围：-1.7976931348623157E+308 到 -2.2250738585072014E-308、0 和 2.2250738585072014E-308 到 1.7976931348623157E+308）
* DECIMAL: 对于 MySQL 5.7 及以上版本，DECIMAL 最多可以存储 65 个数字（范围：-10^65 + 1 到 10^65 - 1）

日期和时间类型（Date and Time Types）
`````````````````````````````````````````````
* DATE: 日期（格式：'YYYY-MM-DD'）
* TIME: 时间（格式：'HH:MM:SS'）
* DATETIME: 日期和时间（格式：'YYYY-MM-DD HH:MM:SS'）
* TIMESTAMP: 时间戳（格式：'YYYY-MM-DD HH:MM:SS'）
* YEAR: 年份（格式：'YYYY'）


**TIMESTAMP 和 DATETIME 都是 MySQL 中的日期和时间类型，但它们有一些不同之处**：

* 存储范围不同：DATETIME 类型存储的日期时间范围为 '1000-01-01 00:00:00' 到 '9999-12-31 23:59:59'，而 TIMESTAMP 类型存储的日期时间范围为 '1970-01-01 00:00:01' UTC 到 '2038-01-19 03:14:07' UTC。
* 存储空间不同：DATETIME 类型需要 8 个字节存储，而 TIMESTAMP 类型只需要 4 个字节。
* 存储方式不同：DATETIME 类型以固定长度存储，需要 8 个字节，其中每个部分都占用一定的字节长度；而 TIMESTAMP 类型以从 1970 年 1 月 1 日起的秒数存储，并且只需要 4 个字节。因此，TIMESTAMP 类型在存储时可以使用较小的空间。
* 默认值不同：DATETIME 类型没有默认值，而 TIMESTAMP 类型的默认值为当前时间。如果将 TIMESTAMP 列设置为 NULL，则会将其值设置为当前时间。
* 自动更新的能力不同：TIMESTAMP 类型可以自动更新，当插入新行时会自动更新为当前时间，但是 DATETIME 类型不会自动更新。

综上所述，如果您需要存储大范围的日期和时间，可以使用 DATETIME 类型。如果您需要节省存储空间或需要自动更新时间戳，则可以使用 TIMESTAMP 类型。注意，由于 TIMESTAMP 类型的范围限制，如果您需要存储更早或更晚的日期时间，则需要使用 DATETIME 类型。

字符串类型（String Types）
`````````````````````````````````````
* CHAR: 固定长度字符串，最多 255 字符
* VARCHAR: 变长字符串，最多 65535 字符
* TINYTEXT: 短文本字符串，最多 255 字符
* TEXT: 文本字符串，最多 65535 字符
* MEDIUMTEXT: 中等长度文本字符串，最多 16777215 字符
* LONGTEXT: 长文本字符串，最多 4294967295 字符

二进制数据类型（Binary Data Types）
`````````````````````````````````````````````````
BLOB类型可以存储二进制数据，如图片、音频、视频等。与字符串类型不同，BLOB类型不会被MySQL自动进行字符集的转换，因为它们是二进制数据。

* BINARY: 固定长度二进制数据，最多 255 字节
* VARBINARY: 变长二进制数据，最多 65535 字节
* TINYBLOB: 短二进制数据，最多 255 字节
* BLOB: 二进制数据，最多 65535 字节
* MEDIUMBLOB: 中等长度二进制数据，最多 16777215 字节
* LONGBLOB: 长二进制数据，最多 4294967295 字节

BINARY一般用于存储小数据，BLOB用于存储较大的数据

枚举类型（Enum Types）
```````````````````````````````````
ENUM: 可以选择一组预定义的值中的一个（最多允许65535个不同的成员）

枚举类型用于存储一个有限的可能值集合中的一个。在创建表时，您可以为 ENUM 列指定一个值列表。例如：

.. code-block:: sql

    CREATE TABLE example (
        id INT PRIMARY KEY,
        color ENUM('red', 'green', 'blue')
    );

上面的语句创建了一个名为 example 的表，其中包含一个名为 color 的 ENUM 列，它可以接受 'red'、'green' 或 'blue' 中的一个。

集合类型（Set Types）
`````````````````````````````````
SET: 可以选择一组预定义的值中的多个（最多允许64个不同的成员）
集合类型与枚举类型类似，不同之处在于集合类型可以接受多个预定义值。在创建表时，您可以为 SET 列指定一个值列表。例如：

.. code-block:: sql

    CREATE TABLE example (
        id INT PRIMARY KEY,
        colors SET('red', 'green', 'blue')
    );

上面的语句创建了一个名为 example 的表，其中包含一个名为 colors 的 SET 列，它可以接受 'red'、'green'、'blue' 中的一个或多个值。

几何类型（Geometry Types）
`````````````````````````````````````
* GEOMETRY: 表示任何几何类型
* POINT: 表示一个二维的坐标点
* LINESTRING: 表示一个或多个连接的线段
* POLYGON: 表示一个封闭的多边形区域
* MULTIPOINT: 表示多个点
* MULTILINESTRING: 表示多个线段
* MULTIPOLYGON: 表示多个多边形区域

几何类型用于存储空间数据，如点、线、面等。在MySQL中，几何类型可以使用空间数据函数进行处理和操作。
例如，您可以使用 ST_Distance_Sphere 函数计算两个点之间的球面距离：

.. code-block:: sql

    SELECT ST_Distance_Sphere(point1, point2) AS distance FROM mytable;

JSON 类型（JSON Data Type）
`````````````````````````````````````
JSON: 存储 JSON 格式的文本数据

JSON 类型用于存储 JSON 格式的文本数据。
JSON 数据可以存储在 VARCHAR、TEXT 或 BLOB 列中，但是如果您需要对其中的 JSON 数据进行操作，建议使用 JSON 数据类型。例如：

.. code-block:: sql

    CREATE TABLE example (
        id INT PRIMARY KEY,
        data JSON
    );


上面的语句创建了一个名为 example 的表，其中包含一个名为 data 的 JSON 列。

UUID 类型（UUID Data Type）
`````````````````````````````````````
UUID: 存储通用唯一标识符（UUID）

UUID 类型用于存储通用唯一标识符（UUID），它是一个128位的二进制数，通常表示为32个十六进制数字，每个数字占4位。
在MySQL中，可以使用 UUID() 函数生成 UUID 值。例如：

.. code-block:: sql

    CREATE TABLE example (
        id INT PRIMARY KEY,
        uuid UUID
    );

上面的语句创建了一个名为 example 的表，其中包含一个名为 uuid 的 UUID 列。


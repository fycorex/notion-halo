---
categories: Notion-Halo
publish: true
excerpt: ''
autoExcerpt: true
tags:
  - Notion
status: 已发布
pinned: ''
public: true
urlname: notion-notice
title: Notion 写作注意事项
date: '2026-04-13 00:07:00'
updated: '2026-04-13 00:07:00'
---

# 前言


在使用 Elog 同步Notion 上的文档时，因为是将富文本向下转成 markdown，会有很多样式损失。这是由于 markdown 样式集合＜ Notion 样式集合。所以在 Notion 上书写时，得按照 markdown 支持的样式进行写作。

> 可以在[这里](/archives/notion-example) 看到 Notion 文档被导出为 markdown 时的样式损失程度

如果你不能接受样式损失，可能 markdown 并不适合你，隔壁 [NotionNext](https://github.com/tangly1024/NotionNext) 可能更适合你搭建博客。


# Notion 格式注意点


### 不要使用 markdown 不支持的样式/语法


例如字体颜色、多级折叠块、书签、数据库、嵌入等。导出为 markdown 都不能正常展示。


### 适当使用 markdown 形式的超链接


在文档中使用markdown 形式的超链接可以解决部分路由问题，例如链接Notion文档的超链接会被自动处理为非完整路径，或者手动链接到某个相对路由，可以使用以下方式解决


```plain text
// 使用[]() markdown 超链接语法
点击 [下一篇](/notion/deploy-platform) 继续配置部署平台
```


### 请勿上传视频、文件到 Notion 文档


Elog 还暂不支持将Notion 中的视频、文件暂不支持上传到图床。如果下载到本地，短期内能访问，但因为 notion 的链接具有时效性，一般是一个小时，之后就不能查看了。


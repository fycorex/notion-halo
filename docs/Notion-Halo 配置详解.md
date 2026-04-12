---
categories: Notion-Halo
publish: true
excerpt: ''
autoExcerpt: true
tags:
  - Elog
status: 已发布
pinned: ''
public: true
urlname: notion-halo-config
title: Notion-Halo 配置详解
date: '2026-04-13 00:07:00'
updated: '2026-04-13 01:32:00'
---

# 前言


参考 [Elog 文档](https://elog.1874.cool/)，仓库中有两个 elog 配置文件：

1. `elog.config.js`：同步到 Halo 时的配置文件，用于同步 Notion 文档到 Halo

    ```javascript
    module.exports = {
      write: {
        platform: 'notion',
        notion: {
          token: process.env.NOTION_TOKEN,
          databaseId: process.env.NOTION_DATABASE_ID,
          filter: { property: 'status', select: { equals: '已发布' }}
        }
      },
      deploy: {
        platform: 'halo',
        halo: {
          endpoint: process.env.HALO_ENDPOINT,
          token: process.env.HALO_TOKEN,
          policyName: process.env.HALO_POLICY_NAME,
          needUploadImage: true
        }
      },
      image: {
        enable: false,
      }
    }
    ```

2. `elog.config.local.js`：同步到本地时的配置文件，用于备份 Notion 文档到本地

    ```javascript
    module.exports = {
      write: {
        platform: 'notion',
        notion: {
          token: process.env.NOTION_TOKEN,
          databaseId: process.env.NOTION_DATABASE_ID,
          filter: { property: 'status', select: { equals: '已发布' }}
        }
      },
      deploy: {
        platform: 'local',
        local: {
          outputDir: './docs',
          filename: 'title',
          format: 'markdown',
          frontMatter: {
            enable: true,
            exclude: ['cover']
          }
        }
      },
      image: {
        enable: true,
        platform: 'local',
        local: {
          outputDir: './images',
          pathFollowDoc: true,
        }
      }
    }
    ```


# Notion 数据库字段

- `urlname`：文章在 Halo 中的唯一 urlname
- `excerpt`：文章描述
- `categories`：文章分类，支持多选
- `tags`：文章标签
- `cover`：文章封面图
- `status`：文章发布状态
- `autoExcerpt`：是否自动生成文章描述
- `publish`：是否发布到 Halo 站点
- `pinned`：是否置顶
- `public`：是否公开

# elog.config.js


## Notion 配置


```javascript
write: {
    platform: 'notion',
    notion: {
      token: process.env.NOTION_TOKEN,
      databaseId: process.env.NOTION_DATABASE_ID,
      filter: { property: 'status', select: { equals: '已发布' }}
    }
  }
```

- `token`为 Notion Token，可从[此处](https://elog.1874.cool/notion/gvnxobqogetukays#token-1)获取
- `databaseId`为数据库的 ID，可从[此处](https://elog.1874.cool/notion/gvnxobqogetukays#databaseid)获取
- `filter`表示 Elog 将下载 notion 数据库属性为`status=已发布`的文档

## Halo 配置


```javascript
deploy: {
  platform: 'halo',
  halo: {
    endpoint: process.env.HALO_ENDPOINT,
    token: process.env.HALO_TOKEN,
    policyName: process.env.HALO_POLICY_NAME,
    needUploadImage: true,
  }
}
```

- `endpoint`表示 Halo 站点地址
- `token`表示 Halo 个人令牌
- `policyName`表示附件的存储策略
- `needUploadImage`表示文档将会把扫描到的图片上传到 Halo中去

## 图床配置


```javascript
image: {
  enable: false
}
```


不使用图床。不推荐图床和`needUploadImage`一起打开，要么用在线图床图片，要么上传到 Halo 中


# elog.config.local.js


## 本地配置


```javascript
deploy: {
  platform: 'local',
  local: {
    outputDir: './docs',
    filename: 'title',
    format: 'markdown',
    frontMatter: {
      enable: true
    }
  }
}
```

- `outputDir`为导出文件夹，文档将被导出到此目录下
- `filename`为文件命名方式，`title` 表示按照文档标题命名
- `format`表示文档将被格式化为 markdown 形式的文档
- `frontMatter.enable`表示将生成带有 FrontMatter 的 markdown 文档

## 图床配置


```javascript
image: {
  enable: true,
  platform: 'local',
  local: {
    outputDir: './images',
    pathFollowDoc: true,
  }
}
```

- `outputDir`为导出文件夹，图片将被导出到此目录下
- `pathFollowDoc`表示文档图片路径将跟随文档路径计算

## 更多 Elog 配置详情，请阅读 [Elog 文档](https://elog.1874.cool/)


![](https://user-images.githubusercontent.com/47793918/233812617-beab2e71-57b9-479e-8bff-c3931347ca40.png)

# C3XL Accord 分发分支

本仓库的 `c3xl-accord` 分支分发 `c3xl-dev` 当前源码与受跟踪的 ARM 运行文件，车辆代码基于 `796e5921e382fae827430b5a9faa26bb7ac631ce`。安装前请阅读[本地修改与适用条件](CHANGELOG.zh-CN.md)；该清单仅列本地增量。

当前配置面向既有 C3XL／雅阁使用场景，包含停用驾驶员摄像头监控、关闭安全带事件检查和 GPIO 蜂鸣器等专用设置，不能据上游介绍推定为所有 comma 硬件或车型的通用安装包。原有 mr-one API／Athena 服务、自动接受条款／培训和上海时区配置仍保留。

**在设备安装界面选择自定义软件，填写：**

```text
installer.comma.ai/bcao812/c3xl-accord
```

安装服务会从本仓库克隆指定分支。不要在该输入框填写普通 GitHub 网页或 ZIP 下载地址。

**下载完整代码并保留版本／更新信息：**

```bash
git clone --depth 1 --branch c3xl-accord https://github.com/bcao812/openpilot.git
```

仓库包含模型分片、预编译文件和随树依赖，没有需要额外初始化的 Git 子模块。手工部署应保留克隆生成的 `.git`。启动环境要求 AGNOS 19.7；需要升级系统时，启动流程仍依赖外部 AGNOS 镜像下载。

[下载源码 ZIP](https://github.com/bcao812/openpilot/archive/refs/heads/c3xl-accord.zip) 可用于查看或保存源码。ZIP 不含 `.git`，当前仓库也不提供替代版本元数据的 `build.json`，请使用上述 Git clone 或设备安装入口完成可更新的安装。

本次源码已有相同车辆代码的主机及 ARM 启动相关测试记录，发布前另核对了受跟踪产物的架构和文件完整性。新设备全量安装、真实相机冷启动和道路验证仍为 **NOT RUN**。既有设备测试保留了本机二进制差异（包括 `liblive.so`），不能等同于此 Git 快照已在新设备完整验证。

---

以下保留上游项目介绍与归属信息；上游的车型和硬件范围不代表本分支的验证范围。


## 🌞 What is sunnypilot?
[sunnypilot](https://github.com/sunnyhaibin/sunnypilot) is a fork of comma.ai's openpilot, an open source driver assistance system. sunnypilot offers the user a unique driving experience for over 300+ supported car makes and models with modified behaviors of driving assist engagements. sunnypilot complies with comma.ai's safety rules as accurately as possible.

## 💭 Join our Community Forum
Join the official sunnypilot community forum to stay up to date with all the latest features and be a part of shaping the future of sunnypilot!
* https://community.sunnypilot.ai/

## Documentation
https://docs.sunnypilot.ai/ is your one stop shop for everything from features to installation to FAQ about the sunnypilot

## 🚘 Running on a dedicated device in a car
First, check out this list of items you'll need to [get started](https://community.sunnypilot.ai/t/getting-started-using-sunnypilot-in-your-supported-car/251).

## Installation
Next, refer to the sunnypilot community forum for [installation instructions](https://community.sunnypilot.ai/t/read-before-installing-sunnypilot/254), as well as a complete list of [Recommended Branch Installations](https://community.sunnypilot.ai/t/recommended-branch-installations/235).

## 🎆 Pull Requests
We welcome both pull requests and issues on GitHub. Bug fixes are encouraged.

Pull requests should be against the most current `master` branch.

## 📊 User Data

By default, sunnypilot uploads the driving data to comma servers. You can also access your data through [comma connect](https://connect.comma.ai/).

sunnypilot is open source software. The user is free to disable data collection if they wish to do so.

sunnypilot logs the road-facing camera, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using this software, you understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.

## Licensing

sunnypilot is released under the [MIT License](LICENSE). This repository includes original work as well as significant portions of code derived from [openpilot by comma.ai](https://github.com/commaai/openpilot), which is also released under the MIT license with additional disclaimers.

The original openpilot license notice, including comma.ai’s indemnification and alpha software disclaimer, is reproduced below as required:

> openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.
>
> Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.
>
> **THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
> YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
> NO WARRANTY EXPRESSED OR IMPLIED.**

For full license terms, please see the [`LICENSE`](LICENSE) file.

## 💰 Support sunnypilot
If you find any of the features useful, consider becoming a [sponsor on GitHub](https://github.com/sponsors/sunnyhaibin) to support future feature development and improvements.


By becoming a sponsor, you will gain access to exclusive content, early access to new features, and the opportunity to directly influence the project's development.


<h3>GitHub Sponsor</h3>

<a href="https://github.com/sponsors/sunnyhaibin">
  <img src="https://user-images.githubusercontent.com/47793918/244135584-9800acbd-69fd-4b2b-bec9-e5fa2d85c817.png" alt="Become a Sponsor" width="300" style="max-width: 100%; height: auto;">
</a>
<br>

<h3>PayPal</h3>

<a href="https://paypal.me/sunnyhaibin0850" target="_blank">
<img src="https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif" alt="PayPal this" title="PayPal - The safer, easier way to pay online!" border="0" />
</a>
<br></br>

Your continuous love and support are greatly appreciated! Enjoy 🥰

<span>-</span> Jason, Founder of sunnypilot

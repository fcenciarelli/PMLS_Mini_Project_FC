

int samples_testing[10][196]= { \
{ \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8094a5a0,  \
0x84808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x881f584c, 0xf3968180,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0xf37e7f7f, 0x7f4ba480, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808084, 0x4d7e5e14, 0x7b7a4488, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x808080ae, 0x7632a485, 0xe47d7bb2, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x808080fd, 0x7ef28480, 0x95697f2c, 0x83808080, 0x80808080, 0x80808080, 0x80808080,  \
0x8080800d, 0x76ae8080, 0x832c7c69, 0x95808080, 0x80808080, 0x80808080, 0x80808080, 0x80809569,  \
0x2d838080, 0x80b35f7b, 0xa5808080, 0x80808080, 0x80808080, 0x80808080, 0x8080b27b, 0xff808080,  \
0x80a55a7b, 0xa5808080, 0x80808080, 0x80808080, 0x80808080, 0x8084fd7f, 0xff808080, 0x80a55a7b,  \
0xa5808080, 0x80808080, 0x80808080, 0x80808080, 0x8084ff7f, 0xff808080, 0x80a95a7b, 0xa5808080,  \
0x80808080, 0x80808080, 0x80808080, 0x8085027f, 0xff808080, 0x88056d76, 0xa0808080, 0x80808080,  \
0x80808080, 0x80808080, 0xa0ec547f, 0x028489b2, 0x307e7f20, 0x83808080, 0x80808080, 0x80808080,  \
0x80808082, 0x297e7f7f, 0x6d5a5f7b, 0x7f773094, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0xda787f7f, 0x7f7f7f77, 0x4dcf8880, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8a517f7f,  \
0x6d5a58ff, 0xa2838080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80027b7f, 0x02848480,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x800d7b7f, 0xff808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x82377e7f, 0x02808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x800c7b7f, 0x4c848080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80027b7f, 0x5a848080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x81207c7f, 0x5a848080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80ff7b7f, 0x5a848080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80f2767f,  \
0x39838080, 0x80808080, 0x80808080, 0x80808080}, \
{ \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x84848080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x808080a2, 0xf1f2b3a5,  \
0x95838080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8080932f, 0x74765f58, 0x2bcf8a80,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8081b379, 0x7f7d7d7f, 0x7c5dffa0, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x8096207f, 0x78342f5d, 0x787f7304, 0x97808080, 0x80808080,  \
0x80808080, 0x80808080, 0x82d1697f, 0x5fb596b4, 0x24757e6b, 0x20898080, 0x80808080, 0x80808080,  \
0x80808080, 0x84ff7b7f, 0x5aa58080, 0x87bb4478, 0x7e1f9480, 0x80808080, 0x80808080, 0x80808080,  \
0x84ff7b7f, 0x5aa58080, 0x8085af32, 0x7d5db280, 0x80808080, 0x80808080, 0x80808080, 0x84f3777f,  \
0x5aa58080, 0x808084f3, 0x7677ff88, 0x80808080, 0x80808080, 0x80808080, 0x83df6f7f, 0x5aa58080,  \
0x808080b3, 0x5f7f4ba0, 0x80808080, 0x80808080, 0x80808080, 0x84fd7a7f, 0x5aa58080, 0x818081b3,  \
0x5f7c2b95, 0x80808080, 0x80808080, 0x80808080, 0x84ff7b7f, 0x5fc5a5a7, 0xc5b1a906, 0x7769d383,  \
0x80808080, 0x80808080, 0x80808080, 0x84ff7b7f, 0x7a5f5a5a, 0x655e5a6c, 0x7d209680, 0x80808080,  \
0x80808080, 0x80808080, 0x84fd7a7f, 0x7f7d765f, 0x6a75510d, 0xf1888080, 0x80808080, 0x80808080,  \
0x80808080, 0x82d26a7f, 0x7d40f4b4, 0xd2f0ae89, 0x84808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80a75a7f, 0x7b028880, 0x82838080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80a55a7f,  \
0x7bff8480, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80a55a7f, 0x7bff8480,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80a55a7f, 0x7bff8480, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80a5587f, 0x7bff8480, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80890d7f, 0x7b0c8880, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x8084fd7f, 0x7c209080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x8082d27d, 0x7b028580, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x8080a676, 0x76fd8480, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x808092fd,  \
0xfdbf8280, 0x80808080, 0x80808080, 0x80808080}, \
{ \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808082, 0x82808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x808080cb, 0xcb888080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808458, 0x74f18080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x8080845a, 0x7bff8080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x8080956a, 0x7afd8080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x8080a67a, 0x6ad28080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8087df7d,  \
0x5bab8484, 0x84848280, 0x80808080, 0x80808080, 0x80808080, 0x80808094, 0xf2126b7f, 0x7a5f5a5a,  \
0x5a581f88, 0x80808080, 0x80808080, 0x80808080, 0x808083ed, 0x767b7f7f, 0x7f7f7b7b, 0x7b7f76cd,  \
0x82808080, 0x80808080, 0x80808080, 0x808083ec, 0x2d166d7d, 0x3e01b3a5, 0xb3407bd4, 0x82808080,  \
0x80808080, 0x80808080, 0x80808092, 0x98d86b6a, 0xd4868080, 0x84307bd2, 0x82808080, 0x80808080,  \
0x80808080, 0x80808080, 0xa2307c5a, 0xa7808080, 0xa2697cd0, 0x82808080, 0x80808080, 0x80808080,  \
0x80808080, 0xfd7a7f6b, 0xe08b8bcf, 0x3c7e5188, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x0d7b7f7e, 0x5e0e0e5d, 0x7c75fe80, 0x80808080, 0x80808080, 0x80808080, 0x80808089, 0x5d7f785b,  \
0x6a7a7c7b, 0x69da8880, 0x80808080, 0x80808080, 0x80808080, 0x808080ae, 0x777c31a9, 0xd2fd200a,  \
0xd2878080, 0x80808080, 0x80808080, 0x80808080, 0x808084f2, 0x7e6ad480, 0x82849088, 0x82808080,  \
0x80808080, 0x80808080, 0x80808080, 0x8080962e, 0x7c2b9580, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x8087df69, 0x69d38380, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80a6597f, 0x0d898080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x81e06b7e, 0xf2848080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x965b7e75,  \
0xae808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8080a0ff, 0x327e69df, 0x81808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080}, \
{ \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80859293, 0x81808080,  \
0x80808484, 0x84828080, 0x80808080, 0x80808080, 0x80808080, 0x8094bfca, 0x87808080, 0x8187a0a5,  \
0xa5948780, 0x80808080, 0x80808080, 0x80808080, 0x82d12a48, 0xa0848388, 0xafdf4b5a, 0x581fda87,  \
0x80808080, 0x80808080, 0x80808080, 0x84f24c57, 0xa58596ae, 0x0130747b, 0x7a5b2096, 0x83808080,  \
0x80808080, 0x80808080, 0x84ff595a, 0xb0a82050, 0x767b6b5b, 0x5f7567d1, 0x95808080, 0x80808080,  \
0x80808080, 0x84ff5a6c, 0x2134767d, 0x6950dfa9, 0xb53266fd, 0xa5808080, 0x80808080, 0x80808080,  \
0x84ff5a7a, 0x646c7d76, 0x30ffa286, 0x8d0e5aff, 0xa5808080, 0x80808080, 0x80808080, 0x85025a7f,  \
0x7f7f6b32, 0xa48a8081, 0x8b0e5aff, 0xa5808080, 0x80808080, 0x80808080, 0x890d5f7f, 0x7f7f5f0e,  \
0x8b818089, 0xa43266fd, 0xa5808080, 0x80808080, 0x80808084, 0xa24d767f, 0x7f7f5f0e, 0x8d858bdb,  \
0x246e5acd, 0x94808080, 0x80808080, 0x80808095, 0xd26a7d7f, 0x7d7c775f, 0x0e020e5d, 0x7330df88,  \
0x81808080, 0x80808080, 0x808080a0, 0xf2767f7f, 0x6d647777, 0x5f5a5e75, 0x72ffaf81, 0x80808080,  \
0x80808080, 0x808080a5, 0xff7b7f77, 0x07cf204b, 0x5a5a582b, 0xfea08780, 0x80808080, 0x80808080,  \
0x808080a5, 0xff7b7f6a, 0xd59fcdf1, 0xfffffdd1, 0xae878180, 0x80808080, 0x80808080, 0x808080b3,  \
0x0d7b7f59, 0xa6858284, 0x84848482, 0x80808080, 0x80808080, 0x80808080, 0x808084f2, 0x4c7f7d2d,  \
0x95828080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x808084fd, 0x587f7b0d, 0x89808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8080890d, 0x5f7e76f2, 0x84808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x8082952d, 0x6a7c69d2, 0x82808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x8084a558, 0x7a6a2d96, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x8084a558, 0x794bf384, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x8084a049, 0x6b1fcd82, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x808087cb,  \
0xe3a18880, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808193, 0x9b878180,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080}, \
{ \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x84848480, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x808084ae, 0xf2fff2b2, 0x93808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x81af3f76, 0x7c7b7a75, 0x2faf8080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0xa1247875, 0x3202ff0e, 0x583e8480, 0x80808080, 0x80808080, 0x80808080,  \
0x8d86808a, 0xff7369fe, 0xa285848c, 0xff568980, 0x80808080, 0x80808080, 0x80808080, 0xf61fa41f,  \
0x6e24a480, 0x80808080, 0x972aa580, 0x80808080, 0x80808080, 0x80808080, 0xff70095b, 0x31a48180,  \
0x80808080, 0xa32fa580, 0x80808080, 0x80808080, 0x80808080, 0x027a6d70, 0xf3848080, 0x80808084,  \
0xf370a580, 0x80808080, 0x80808080, 0x80808083, 0x4b7f7f5a, 0xa7808080, 0x808087db, 0x5e79a580,  \
0x80808080, 0x80808080, 0x80808082, 0x2b7c7f5a, 0xa5808080, 0x8080b050, 0x7e6a9580, 0x80808080,  \
0x80808080, 0x80808080, 0xd46a7f5a, 0xa5808084, 0x89ae3178, 0x7e4b8480, 0x80808080, 0x80808080,  \
0x80808080, 0x952d7d6d, 0x13ff024d, 0x5f777e75, 0x24a38080, 0x80808080, 0x80808080, 0x80808080,  \
0x85027b7f, 0x7b7b7b7e, 0x7d764dfe, 0xa1818080, 0x80808080, 0x80808080, 0x80808080, 0x8b0e7b6d,  \
0x5a5a5a58, 0x2df3a288, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0xcd5d7f13, 0xa9a5a5a5,  \
0x95848080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0xff7b7fff, 0x84808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0xff7b7fff, 0x84808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x0d7b7fff, 0x84808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808084, 0x587f7ff2, 0x84808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808084, 0x5a7f7bb3, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808084,  \
0x5a7f7ba5, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808095, 0x6a7f6995,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x808080a0, 0x747d2d83, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080}, \
{ \
0x80808080, 0x80808489, 0xa0d2fdff, 0xfffdd2a2, 0x81808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80a14b5e, 0x72745e5a, 0x5f7a7974, 0x0cd29683, 0x80808080, 0x80808080, 0x80808080, 0x8cff7d5f,  \
0x0ef2b3a5, 0xb3fd0d51, 0x76682ccf, 0x81808080, 0x80808080, 0x80808094, 0xfb5f7c1b, 0x92848080,  \
0x808489ae, 0x0c51755b, 0x96808080, 0x80808080, 0x808081c5, 0x657f7f55, 0xa4808080, 0x80808080,  \
0x808c0c79, 0x20948080, 0x80808080, 0x808080a6, 0x597f6dff, 0x92808080, 0x80808080, 0x8082c23c,  \
0x5ebf8580, 0x80808080, 0x80808094, 0x20785fb6, 0x81808080, 0x80808080, 0x808084d4, 0x7744ad80,  \
0x80808080, 0x80808080, 0x962e7c2b, 0x95808080, 0x80808080, 0x808080a5, 0x7b6edc80, 0x80808080,  \
0x80808080, 0x85ff7f5d, 0xb4808080, 0x80808080, 0x808083b5, 0x7b64c680, 0x80808080, 0x80808080,  \
0x82d27d78, 0x20838080, 0x80808080, 0x808acf31, 0x77209680, 0x80808080, 0x80808080, 0x8088517f,  \
0x76a28080, 0x83818081, 0xda507878, 0x1b968080, 0x80808080, 0x80808080, 0x80800d7c, 0x7de5a7b2,  \
0xe6c7a7b4, 0x5c7e7725, 0x96818080, 0x80808080, 0x80808080, 0x8080f174, 0x7f6b5b5f, 0x72665a5e,  \
0x755d1fa3, 0x80808080, 0x80808080, 0x80808080, 0x8080a125, 0x787f7f7b, 0x6a5a5a4c, 0xf3b29481,  \
0x80808080, 0x80808080, 0x80808080, 0x808080a6, 0x597d34b5, 0x95848484, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808095, 0x2b7d34a3, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808083, 0xd37c75fe, 0x88808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x895d7f58, 0xa5808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x822d7d5a, 0xa7808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80f3766a,  \
0xd4808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80a24d7e, 0x51888080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80890c7b, 0x76a08080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8082d269, 0x7aa58080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x808087cd, 0xef878080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808082, 0x83808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080}, \
{ \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808082, 0x82808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x8088cd2d, 0x2bce8080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x8924767f, 0x7f6bcf87, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808087, 0xda767f6d, 0x787f4aa0, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x808080a6,  \
0x597f5fc4, 0x0e7b6ad2, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x808080d4, 0x6a770183,  \
0xff7b75f0, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8081af69, 0x7e209681, 0x0e7b5aa7,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80ae507f, 0x7aa680cd, 0x777f58a5, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80f2767f, 0x6a95830c, 0x7f7e1f94, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80ff7b7e, 0x3cd22c78, 0x78448a80, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x800d7b7f, 0x635f7c7e, 0x25af8080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x85597f7f, 0x7f7f7b24, 0x89808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0xae777f7a,  \
0x52f3b287, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0xf27f7f52, 0xbc848080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0xff7f7f02, 0x85808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x0d7f7fff, 0x84808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808084, 0x4c7f7ff2, 0x84808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808089, 0x5f7f6a96, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x808080a2, 0x767f5a85, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x808084f2,  \
0x7f7d2d82, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x808084ff, 0x7f76f280,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8080890d, 0x7f5fb280, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8080a558, 0x7f2d9580, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x8080a55a, 0x7f028580, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x8080941a, 0x4cc18280, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808082, 0x84808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080}, \
{ \
0x80808080, 0x80808080, 0x80808084, 0x95a5a5a0, 0x88808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x808aa2f3, 0x2d58584c, 0xfea08080, 0x80808080, 0x80808080, 0x80808283, 0x80808083,  \
0xa2ff3074, 0x7b7b7b7c, 0x73ff8a81, 0x80808080, 0x80808080, 0x8082c1ea, 0x928283c3, 0x2f747a74,  \
0x3202123e, 0x7c5dcf96, 0x80808080, 0x80808080, 0x8088f936, 0xa0848cfb, 0x67756a4b, 0xdfa7c006,  \
0x7775feae, 0x80808080, 0x80808080, 0x84a24d59, 0xa899fb5f, 0x6a05d4a2, 0x878089b3, 0x5f7e4cf2,  \
0x84808080, 0x80808080, 0x95d26a5f, 0xd5e86876, 0x20978480, 0x808084a5, 0x5a7f5aff, 0x84808080,  \
0x80808080, 0xb20c7b7a, 0x646c7524, 0xa3818080, 0x808084a5, 0x5a7f5aff, 0x84808080, 0x80808080,  \
0xf24c7f7f, 0x7f7825a3, 0x81808080, 0x80808bb4, 0x5f7f58fd, 0x84808080, 0x80808080, 0xfd587f7f,  \
0x7d6be089, 0x80808080, 0x8080a2df, 0x6b7e4cf2, 0x84808080, 0x80808080, 0xff5a7f7f, 0x764da280,  \
0x80808080, 0x8084f34b, 0x7e77ffae, 0x80808080, 0x80808080, 0xff5a7f7f, 0x5f0d8980, 0x80808080,  \
0x80962d6a, 0x7f51ae88, 0x80808080, 0x80808080, 0xff5a7f7f, 0x5aff8480, 0x80808081, 0x96e4697c,  \
0x76ff8880, 0x80808080, 0x80808080, 0xff5a7f7f, 0x5a018da0, 0xa19aa1b4, 0x20697d7a, 0x44ad8080,  \
0x80808080, 0x80808080, 0xff5a7f7f, 0x5d15ff4c, 0x4e3a4e5f, 0x787e7350, 0xbb858080, 0x80808080,  \
0x80808080, 0xff5a7f7f, 0x632d4876, 0x776f777b, 0x7f784f1b, 0x97808080, 0x80808080, 0x80808080,  \
0xff5a7f7f, 0x7a707a7f, 0x7f7f7f7e, 0x6920af96, 0x81808080, 0x80808080, 0x80808080, 0xff5a7f7f,  \
0x62456b7e, 0x7d7a7650, 0xdf968180, 0x80808080, 0x80808080, 0x80808080, 0xfd587f7c, 0x12c9fb4b,  \
0x2dfff2ae, 0x87808080, 0x80808080, 0x80808080, 0x80808080, 0xd22d7d7d, 0x2dd38ba0, 0x95858480,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0xb20d7b7f, 0x4cf28483, 0x82808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0xa0f1747c, 0x4af18480, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x87a11d29, 0xcea08080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80819394, 0x83808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080}, \
{ \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x84848482, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808180, 0x808087ae,  \
0xf2fffdcd, 0x8a808080, 0x80808080, 0x80808080, 0x80808080, 0x81a0c494, 0x8396df4f, 0x7276785d,  \
0xffa08080, 0x80808080, 0x80808080, 0x80808080, 0x941e6001, 0xd42c665c, 0x0eff2e69, 0x72ff8a80,  \
0x80808080, 0x80808080, 0x80808080, 0xa55a7f7d, 0x7750df96, 0x818083b4, 0x5d75fe80, 0x80808080,  \
0x80808080, 0x80808080, 0xa04d7f7f, 0x50bb8780, 0x80808094, 0x20774c84, 0x80808080, 0x80808080,  \
0x80808080, 0x8e1b7c7d, 0xdf878080, 0x80808081, 0xaf515a85, 0x80808080, 0x80808080, 0x80808080,  \
0xa24d7f7b, 0xa9808080, 0x80808080, 0x890d6695, 0x80808080, 0x80808080, 0x8080808a, 0x01777f7f,  \
0xf2848080, 0x80808080, 0x84ff76a5, 0x80808080, 0x80808080, 0x808080cd, 0x5d78727f, 0x0d898080,  \
0x80808080, 0x8b0e72a0, 0x80808080, 0x80808080, 0x808080ff, 0x7735227c, 0x4da48480, 0x8080848b,  \
0xdb5d5b89, 0x80808080, 0x80808080, 0x8080822d, 0x68d9037b, 0x6c11f2b3, 0xa5a7f20e, 0x5c5cfa82,  \
0x80808080, 0x80808080, 0x80808458, 0x5aacff7b, 0x6e416b5a, 0x55567172, 0x4fdb8a80, 0x80808080,  \
0x80808080, 0x8080845a, 0x5aa8d26a, 0x5aada0a5, 0xa5a5a5a0, 0x88808080, 0x80808080, 0x80808080,  \
0x8080845a, 0x58a5a75a, 0x5aa58080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808457,  \
0x2e96a55a, 0x5aa58080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808558, 0x2e96a55a,  \
0x5aa58080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80809065, 0x58a5a55a, 0x5aa58080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8080832d, 0x58a7a65a, 0x5aa58080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808002, 0x66e0ad5a, 0x5aa58080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x808080ff, 0x7950d55b, 0x5aa58080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x808080ed, 0x6978456e, 0x58a58080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808084, 0xd4697f7f, 0xff888080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x961b6972, 0xae808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8196d1ef,  \
0x87808080, 0x80808080, 0x80808080, 0x80808080}, \
{ \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808280, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x808088a0, 0xa5b2d0a2, 0x88808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x83a2fe4c,  \
0x5a5f6a4d, 0xffb3a594, 0x81808080, 0x80808080, 0x80808080, 0x80808083, 0xc330757f, 0x7f7f7d7e,  \
0x765f581b, 0x96808080, 0x80808080, 0x80808080, 0x808080a2, 0x307b7f71, 0x725d2f59, 0x6b7e7f69,  \
0xdf888080, 0x80808080, 0x80808080, 0x808088fe, 0x757f7312, 0xe9cd97a6, 0xe05c7e7e, 0x5bcd8280,  \
0x80808080, 0x80808080, 0x8080a24d, 0x7e6b04a1, 0x84828080, 0x89db5c7e, 0x7afd8480, 0x80808080,  \
0x80808080, 0x8083d46a, 0x7d2e9880, 0x80808080, 0x8089db5e, 0x7b0e8b80, 0x80808080, 0x80808080,  \
0x80ae517f, 0x6ad28280, 0x80808080, 0x808085ff, 0x7a74f184, 0x80808080, 0x80808080, 0x84f2767f,  \
0x5ba98080, 0x80808080, 0x808082d2, 0x6a7b0d89, 0x80808080, 0x80808080, 0x84ff7b7f, 0x6ad28280,  \
0x80808080, 0x808080a7, 0x5a7e4ca0, 0x80808080, 0x80808080, 0x84ff7b7f, 0x7afd8480, 0x80808080,  \
0x808080a5, 0x5a7e4ca0, 0x80808080, 0x80808080, 0x84f3777f, 0x7bff8480, 0x80808080, 0x808080b3,  \
0x5f7b0c89, 0x80808080, 0x80808080, 0x83df6f7f, 0x7bff8480, 0x80808080, 0x80809605, 0x765dcd82,  \
0x80808080, 0x80808080, 0x84fd7a7f, 0x7b0e8d82, 0x80808080, 0x8087ef68, 0x73ff8a80, 0x80808080,  \
0x80808080, 0x84fd797f, 0x7f5f0cd2, 0xa7a5a5a5, 0xb3f43f67, 0x04a08080, 0x80808080, 0x80808080,  \
0x82c23c7d, 0x7f7f7b6a, 0x5a5a5a5a, 0x5e745bfb, 0x96808080, 0x80808080, 0x80808080, 0x8084d46a,  \
0x7f7f7c7b, 0x7b7b765e, 0x5a4bdb8a, 0x80808080, 0x80808080, 0x80808080, 0x8080a55a, 0x7f5fc5a5,  \
0xa5a5a089, 0x84848080, 0x80808080, 0x80808080, 0x80808080, 0x8080a55a, 0x7f5aa780, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x8080a55a, 0x7f65c581, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x8080a55a, 0x7f5eb180, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x8080a55a, 0x7f5aa580, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x8080a03f, 0x7656a580, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x808085ad, 0xf2ec9280, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x84838080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,  \
0x80808080, 0x80808080, 0x80808080, 0x80808080}, \
};

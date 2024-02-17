// automatically generated by the FlatBuffers compiler, do not modify
import * as flatbuffers from 'flatbuffers';
import { EnumInNestedNS } from '../namespace-a/namespace-b/enum-in-nested-n-s';
import { StructInNestedNS } from '../namespace-a/namespace-b/struct-in-nested-n-s';
import { TableInNestedNS } from '../namespace-a/namespace-b/table-in-nested-n-s';
import { UnionInNestedNS, unionToUnionInNestedNS } from '../namespace-a/namespace-b/union-in-nested-n-s';
export class TableInFirstNS {
    constructor() {
        this.bb = null;
        this.bb_pos = 0;
    }
    __init(i, bb) {
        this.bb_pos = i;
        this.bb = bb;
        return this;
    }
    static getRootAsTableInFirstNS(bb, obj) {
        return (obj || new TableInFirstNS()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
    }
    static getSizePrefixedRootAsTableInFirstNS(bb, obj) {
        bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
        return (obj || new TableInFirstNS()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
    }
    fooTable(obj) {
        const offset = this.bb.__offset(this.bb_pos, 4);
        return offset ? (obj || new TableInNestedNS()).__init(this.bb.__indirect(this.bb_pos + offset), this.bb) : null;
    }
    fooEnum() {
        const offset = this.bb.__offset(this.bb_pos, 6);
        return offset ? this.bb.readInt8(this.bb_pos + offset) : EnumInNestedNS.A;
    }
    mutate_foo_enum(value) {
        const offset = this.bb.__offset(this.bb_pos, 6);
        if (offset === 0) {
            return false;
        }
        this.bb.writeInt8(this.bb_pos + offset, value);
        return true;
    }
    fooUnionType() {
        const offset = this.bb.__offset(this.bb_pos, 8);
        return offset ? this.bb.readUint8(this.bb_pos + offset) : UnionInNestedNS.NONE;
    }
    fooUnion(obj) {
        const offset = this.bb.__offset(this.bb_pos, 10);
        return offset ? this.bb.__union(obj, this.bb_pos + offset) : null;
    }
    fooStruct(obj) {
        const offset = this.bb.__offset(this.bb_pos, 12);
        return offset ? (obj || new StructInNestedNS()).__init(this.bb_pos + offset, this.bb) : null;
    }
    static getFullyQualifiedName() {
        return 'NamespaceA.TableInFirstNS';
    }
    static startTableInFirstNS(builder) {
        builder.startObject(5);
    }
    static addFooTable(builder, fooTableOffset) {
        builder.addFieldOffset(0, fooTableOffset, 0);
    }
    static addFooEnum(builder, fooEnum) {
        builder.addFieldInt8(1, fooEnum, EnumInNestedNS.A);
    }
    static addFooUnionType(builder, fooUnionType) {
        builder.addFieldInt8(2, fooUnionType, UnionInNestedNS.NONE);
    }
    static addFooUnion(builder, fooUnionOffset) {
        builder.addFieldOffset(3, fooUnionOffset, 0);
    }
    static addFooStruct(builder, fooStructOffset) {
        builder.addFieldStruct(4, fooStructOffset, 0);
    }
    static endTableInFirstNS(builder) {
        const offset = builder.endObject();
        return offset;
    }
    unpack() {
        return new TableInFirstNST((this.fooTable() !== null ? this.fooTable().unpack() : null), this.fooEnum(), this.fooUnionType(), (() => {
            let temp = unionToUnionInNestedNS(this.fooUnionType(), this.fooUnion.bind(this));
            if (temp === null) {
                return null;
            }
            return temp.unpack();
        })(), (this.fooStruct() !== null ? this.fooStruct().unpack() : null));
    }
    unpackTo(_o) {
        _o.fooTable = (this.fooTable() !== null ? this.fooTable().unpack() : null);
        _o.fooEnum = this.fooEnum();
        _o.fooUnionType = this.fooUnionType();
        _o.fooUnion = (() => {
            let temp = unionToUnionInNestedNS(this.fooUnionType(), this.fooUnion.bind(this));
            if (temp === null) {
                return null;
            }
            return temp.unpack();
        })();
        _o.fooStruct = (this.fooStruct() !== null ? this.fooStruct().unpack() : null);
    }
}
export class TableInFirstNST {
    constructor(fooTable = null, fooEnum = EnumInNestedNS.A, fooUnionType = UnionInNestedNS.NONE, fooUnion = null, fooStruct = null) {
        this.fooTable = fooTable;
        this.fooEnum = fooEnum;
        this.fooUnionType = fooUnionType;
        this.fooUnion = fooUnion;
        this.fooStruct = fooStruct;
    }
    pack(builder) {
        const fooTable = (this.fooTable !== null ? this.fooTable.pack(builder) : 0);
        const fooUnion = builder.createObjectOffset(this.fooUnion);
        TableInFirstNS.startTableInFirstNS(builder);
        TableInFirstNS.addFooTable(builder, fooTable);
        TableInFirstNS.addFooEnum(builder, this.fooEnum);
        TableInFirstNS.addFooUnionType(builder, this.fooUnionType);
        TableInFirstNS.addFooUnion(builder, fooUnion);
        TableInFirstNS.addFooStruct(builder, (this.fooStruct !== null ? this.fooStruct.pack(builder) : 0));
        return TableInFirstNS.endTableInFirstNS(builder);
    }
}
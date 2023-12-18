import { useLocation, useOutletContext, useParams } from "react-router-dom";
import Card from "../Card";
import ReactPaginate from 'react-paginate';
import { useEffect, useState } from "react";



export default function SearchResult() {
    const [offset, setOffset] = useState(0);
    const [data, setData] = useState([])
    const [perPage] = useState(10);
    const [pageCount, setPageCount] = useState(0)
    const {searchId} = useParams();
    const [docs] = useOutletContext();
    const docsLength = docs.length



    function setPageData(data, selectedPage) {
        let indexOfLastItem;
        let indexOfFirstItem;
        if (selectedPage === undefined || selectedPage === 1|| selectedPage === 201) {
            indexOfLastItem = 1 * perPage;
            indexOfFirstItem = 0;
        }
        else {

            indexOfLastItem = (selectedPage + 1) * perPage;
            indexOfFirstItem = indexOfLastItem - perPage;
        }
        console.log(`selectedPage:${selectedPage} ,indexOfFirstItem:${indexOfFirstItem} ,indexOfLastItem:${indexOfLastItem} `)

        const pageData = data.slice(indexOfFirstItem, indexOfLastItem)
        console.log(`selectedPageData:${pageData}`)
        setData(pageData, selectedPage)
        setPageCount(Math.ceil(docsLength / perPage))
    }

    function handlePageClick(e) {
        const selectedPage = e.selected;
        console.log(`selectedPage:${selectedPage}`)
        console.log('enters')
        setOffset(selectedPage + 1)
        console.log(`offset:${offset}`)
        manipulateData(selectedPage)
    };

    function manipulateData(selectedPage=1) {
        if (parseInt(searchId) === 0) {
            selectedPage = 1
            setPageData(docs, selectedPage)
            return;
        }
        let filteredData = docs.filter((doc) => doc.clusterId == `${searchId}`)
        console.log(searchId)
        console.log(filteredData)
        setPageData(filteredData, selectedPage)
    }

    useEffect(() => {
        //let selectedPage = 0
        manipulateData()
        //const selectedPage = e.selected;
        console.log(`docs:${docs}`)
    }, [searchId])

    return (

        <div className=" flex flex-col  px-40 ml-5">
            {data.map((doc) =>
                <Card key={doc.documentId} docData={doc} searchId={searchId} />
            )}
            <ReactPaginate
                className="flex  text-white space-x-2"
                previousLabel={"prev"}
                nextLabel={"next"}
                breakLabel={"..."}
                breakClassName={"break-me"}
                pageCount={pageCount}
                marginPagesDisplayed={2}
                pageRangeDisplayed={5}
                onPageChange={handlePageClick}
                containerClassName={"pagination"}
                subContainerClassName={"pages pagination"}
                activeClassName={"active"}
                //forcePage={0}
            />
        </div>
    )
}

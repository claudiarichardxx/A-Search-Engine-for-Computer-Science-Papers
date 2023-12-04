import { useLocation, useOutletContext, useParams } from "react-router-dom";
import Card from "../Card";
import ReactPaginate from 'react-paginate';
import { useEffect, useState } from "react";



export default function SearchResult() {
    const [offset, setOffset] = useState(0);
    const [data, setData] = useState([])
    const [perPage] = useState(4);
    const [pageCount, setPageCount] = useState(0)
    const { searchId } = useParams();
    const [docs] = useOutletContext();
    let docsLength= docs.Length




    let indexOfLastItem;
    let indexOfFirstItem;
    function setPageData(data,selectedPage) {
        if(selectedPage===undefined || selectedPage===0 ){
             indexOfLastItem = 1 * perPage;
             indexOfFirstItem = 0;
        }
        else{

            indexOfLastItem = (selectedPage+1) * perPage;
            indexOfFirstItem = indexOfLastItem - perPage;
        }
        console.log(`selectedPage:${selectedPage} ,indexOfFirstItem:${indexOfFirstItem} ,indexOfLastItem:${indexOfLastItem} `)
        
        const pageData = data.slice(indexOfFirstItem,indexOfLastItem)
        setData(pageData, selectedPage)
        setPageCount(Math.ceil(docsLength / perPage))
    }

    function handlePageClick(e) {
        const selectedPage = e.selected;
        console.log(`selectedPage:${selectedPage}`)
        setOffset(selectedPage + 1)
        console.log(`offset:${offset}`)
        manipulateData(selectedPage)
    };

    function manipulateData(selectedPage) {
        if (parseInt(searchId) === 0) {
            setPageData(docs,selectedPage)
            return;
        }
        let filteredData = docs.filter((doc) => doc.clusterId == `${searchId}`)
        console.log(searchId)
        console.log(filteredData)
        setPageData(filteredData,selectedPage)
    }

    useEffect(() => {
        let selectedPage = 0
        manipulateData(selectedPage)
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
            />
        </div>
    )
}